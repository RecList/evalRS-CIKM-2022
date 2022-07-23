"""

    This is the abstract class containing the common methods for the evaluation of your model.
    Your own submission will implement an instance of EvalRSRunner, containing your trainining logic,
    and the reference to your model object (i.e. a Python class exposing a `predict_all` method).

    You should NOT modify this script.

"""

import os
from abc import ABC, abstractmethod
import pandas as pd
import time
import numpy as np
from typing import List
import json
from evaluation.EvalRSRecList import EvalRSRecList, EvalRSDataset
from collections import defaultdict
from evaluation.utils import download_with_progress, get_cache_directory, LFM_DATASET_PATH, decompress_zipfile, upload_submission


class EvalRSRunner(ABC):

    def __init__(self,
                 seed: int = None,
                 num_folds: int = 4,
                 email: str = None,
                 participant_id: str = None,
                 aws_access_key_id: str = None,
                 aws_secret_access_key: str = None,
                 bucket_name: str = None,
                 force_download: bool = False):
        # download dataset
        self.path_to_dataset = os.path.join(get_cache_directory(), 'lfm_1b_dataset')
        if not os.path.exists(self.path_to_dataset) or force_download:
            print("Downloading LFM dataset...")
            download_with_progress(LFM_DATASET_PATH, os.path.join(get_cache_directory(), 'lfm_1b_dataset.zip'))
            decompress_zipfile(os.path.join(get_cache_directory(), 'lfm_1b_dataset.zip'),
                                    get_cache_directory())
        else:
            print("LFM dataset already downloaded. Skipping download.")

        self.path_to_events = os.path.join(self.path_to_dataset, 'LFM-1b_events.pk')
        self.path_to_tracks = os.path.join(self.path_to_dataset, 'LFM-1b_tracks.pk')
        self.path_to_users = os.path.join(self.path_to_dataset, 'LFM-1b_users.pk')
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

    def _generate_folds(self, num_folds: int, seed: int) -> List[pd.DataFrame]:
        df_rand = self._df_events.sample(frac=1.0, replace=False, random_state=seed)
        folds = np.array_split(df_rand, num_folds)
        # some mem management
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
        test_set_df.columns = [str(_) for _ in test_set_df.columns]
        return test_set_df.set_index('user_id')

    def _test_model(self, model, fold: int, limit: int = None) -> str:

        test_set_df = self._get_test_set(fold=fold, limit=limit)
        x_test = test_set_df.reset_index()[['user_id']]
        y_test = test_set_df

        dataset = EvalRSDataset()
        dataset.load(x_test=x_test,
                     y_test=y_test,
                     users=self.df_users,
                     items=self.df_tracks)

        rlist = EvalRSRecList(model=model, dataset=dataset)
        report_path = rlist()
        return report_path

    def evaluate(self, upload: bool, debug=True, limit: int = None):
        if upload:
            assert self.email
            assert self.participant_id
            assert self.aws_access_key_id
            assert self.aws_secret_access_key
            assert self.bucket_name

        num_folds = len(self._folds)
        fold_results_path = []
        for fold in range(num_folds):
            train_df = self._get_train_set(fold=fold)
            if debug:
                print('\nPerforming Training for fold {}/{}...'.format(fold+1, num_folds))
            model = self.train_model(train_df)
            if debug:
                print('Performing Evaluation for fold {}/{}...'.format(fold+1, num_folds))
            results_path = self._test_model(model, fold, limit=limit)

            fold_results_path.append(results_path)

        raw_results = []
        fold_results = defaultdict(list)
        for fold, results_path in enumerate(fold_results_path):
            with open(os.path.join(results_path, 'results', 'report.json')) as f:
                result = json.load(f)
            # save reclist output
            raw_results.append(result)
            # tests which we care about
            tests = ['HIT_RATE']
            # extract test results
            for test_data in result['data']:
                if test_data['test_name'] in tests:
                    fold_results[test_data['test_name']].append(test_data['test_result'])

        # compute means
        # TODO: CI computation
        agg_results = {test: np.mean(res) for test, res in fold_results.items()}
        # build final output dict
        out_dict = {
            'reclist_reports': raw_results,
            'results': agg_results
        }
        # TODO: dump data somewhere better?
        local_file = '{}_{}.json'.format(self.email.replace('@', '_'), int(time.time()*10000))
        with open(local_file, 'w') as outfile:
            json.dump(out_dict, outfile, indent=2)
        print('SUBMISSION RESULTS SAVE TO {}'.format(local_file))
        if upload:
            upload_submission(local_file,
                              aws_access_key_id=self.aws_access_key_id,
                              aws_secret_access_key=self.aws_secret_access_key,
                              participant_id=self.participant_id,
                              bucket_name=self.bucket_name)

    @abstractmethod
    def train_model(self, train_df: pd.DataFrame):
        raise NotImplementedError
