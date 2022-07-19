import os
from abc import ABC, abstractmethod
import pandas as pd
import time
import numpy as np
from typing import List

class EvalRSRunner(ABC):

    def __init__(self, path_to_dataset: str, seed:int = None, num_folds:int = 5):
        self.path_to_dataset = path_to_dataset
        self.path_to_events = os.path.join(path_to_dataset, 'LFM-1b_events.pk')
        self.path_to_tracks = os.path.join(path_to_dataset, 'LFM-1b_tracks.pk')
        self.path_to_users = os.path.join(path_to_dataset, 'LFM-1b_users.pk')

        assert os.path.exists(self.path_to_events)
        assert os.path.exists(self.path_to_tracks)
        assert os.path.exists(self.path_to_users)

        self.df_events = pd.read_parquet(self.path_to_events)
        # self.df_tracks = pd.read_parquet(self.path_to_events)
        # self.df_users = pd.read_parquet(self.path_to_events)
        self._random_state = int(time.time()) if not seed else seed
        self._num_folds = num_folds
        self._folds = self._generate_folds(num_folds, self._random_state)

    def _generate_folds(self, num_folds: int, seed: int) -> (List[pd.DataFrame], int):
        df_rand = self.df_events.sample(frac=1.0, replace=False, random_state=seed)
        return np.array_split(df_rand, num_folds), seed

    def _get_train_set(self, fold: int) -> pd.DataFrame:
        assert fold < len(self._folds)
        return pd.concat([self._folds[idx] for idx in range(len(self._folds)) if idx != fold])

    def _get_test_set(self, fold: int) -> pd.DataFrame:
        return self._folds[fold]


    def _test_model(self, model, fold: int):
        test_df = self._get_test_set(fold=fold)
        # do predictions

    def evaluate(self, upload:bool = False):
        num_folds = len(self._folds)
        fold_results = []
        for fold in range(num_folds):
            train_df = self._get_train_set(fold=fold)
            model = self.train_model(train_df)
            # TODO: call RecList here instead
            # append RecList result path into fold_results
        if upload:
            # TODO: iterate and read from RecList artifacts, and aggregate across folds
            # TODO: Upload results to S3 bucket
            pass

    @abstractmethod
    def train_model(self, train_df: pd.DataFrame):
        raise NotImplementedError




if __name__ == '__main__':
    runner = EvalRSRunner(path_to_dataset='./lfm_1b_dataset')
    df = runner._get_train_set(fold=0)
    print(df.head(10))