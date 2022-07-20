import os
from abc import ABC, abstractmethod
import pandas as pd
import time
import numpy as np
from typing import List
import json
from submission.uploader import upload_submission


class EvalRSRunner(ABC):

    def __init__(self,
                 path_to_dataset: str,
                 seed: int = None,
                 num_folds: int = 5,
                 email: str = None,
                 participant_id: str = None,
                 aws_access_key_id: str = None,
                 aws_secret_access_key: str = None,
                 bucket_name: str = None):

        self.path_to_dataset = path_to_dataset
        self.path_to_events = os.path.join(path_to_dataset, 'LFM-1b_events.pk')
        self.path_to_tracks = os.path.join(path_to_dataset, 'LFM-1b_tracks.pk')
        self.path_to_users = os.path.join(path_to_dataset, 'LFM-1b_users.pk')
        self.email = email
        self.participant_id = participant_id
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.bucket_name = bucket_name

        assert os.path.exists(self.path_to_events)
        assert os.path.exists(self.path_to_tracks)
        assert os.path.exists(self.path_to_users)

        self._df_events = pd.read_parquet(self.path_to_events)
        self.df_tracks = pd.read_parquet(self.path_to_events)
        self.df_users = pd.read_parquet(self.path_to_events)
        self._random_state = int(time.time()) if not seed else seed
        self._num_folds = num_folds
        self._folds = self._generate_folds(num_folds, self._random_state)

    def _generate_folds(self, num_folds: int, seed: int) -> (List[pd.DataFrame], int):
        df_rand = self._df_events.sample(frac=1.0, replace=False, random_state=seed)
        folds = np.array_split(df_rand, num_folds)
        # mem management
        del self._df_events
        return folds

    def _get_train_set(self, fold: int) -> pd.DataFrame:
        assert fold < len(self._folds)
        return pd.concat([self._folds[idx] for idx in range(len(self._folds)) if idx != fold])

    def _get_test_set(self, fold: int, limit: int = None) -> pd.DataFrame:
        if limit:
            print('WARNING : LIMITING TEST EVENTS TO {} EVENTS ONLY'.format(limit))
        # get held-out split
        test_set_events = self._folds[fold] if not limit else self._folds[fold].head(limit)
        # get tracks listened by user
        test_set = (test_set_events[['user_id', 'track_id']]
                    .groupby(by=['user_id'])['track_id']
                    .apply(set)
                    .apply(list))
        # save index
        test_set_index = test_set.index.to_numpy()
        # convert to list of list
        test_set = test_set.tolist()
        # convert to pd.DataFrame; columns are tracks
        test_set_df = pd.DataFrame(test_set).fillna(value=-1).astype(np.int64)
        # set index to original index
        test_set_df['user_id'] = test_set_index
        return test_set_df.set_index('user_id')

    def _test_model(self, model, fold: int, limit:int = None):

        y_test = self._get_test_set(fold=fold, limit=limit)
        # for fast testing
        y_pred = model.predict(y_test.index.to_numpy(),
                          k=100)
        hits = np.stack([(y_test.values == y_pred[col].values.reshape(-1, 1)) for col in y_pred.columns])
        hits = (hits.sum(axis=0) > 0)
        hit_rate = hits.sum() / (y_test != -1).values.sum()
        return {
            'FOLD': fold,
            'HIT_RATE': hit_rate
        }

    def evaluate(self, upload:bool, debug=True, limit:int = None):
        if upload:
            assert self.email
            assert self.participant_id
            assert self.aws_access_key_id
            assert self.aws_secret_access_key
            assert self.bucket_name

        num_folds = len(self._folds)
        fold_results = []
        for fold in range(num_folds):
            train_df = self._get_train_set(fold=fold)
            if debug:
                print('\nPerforming Training for fold {}/{}'.format(fold+1, num_folds))
            model = self.train_model(train_df)
            # TODO: call RecList here instead
            if debug:
                print('Performing Evaluation for fold {}/{}'.format(fold+1, num_folds))
            results = self._test_model(model, fold, limit=limit)
            # append RecList result path into fold_results
            fold_results.append(results)

        local_file = '{}_{}.json'.format(self.email.replace('@', '_'), int(time.time()*10000))
        with open(local_file,'w') as outfile:
            json.dump(fold_results, outfile, indent=2)

        if upload:
            # TODO: iterate and read from RecList artifacts, and aggregate across folds
            upload_submission(local_file,
                              aws_access_key_id = self.aws_access_key_id,
                              aws_secret_access_key = self.aws_secret_access_key,
                              participant_id = self.participant_id,
                              bucket_name = self.bucket_name)

    # @abstractmethod
    def train_model(self, train_df: pd.DataFrame):
        raise NotImplementedError


if __name__ == '__main__':
    runner = EvalRSRunner(path_to_dataset='./lfm_1b_dataset')
    df = runner._get_train_set(fold=0)
    print(df.head(10))