"""

    This is the class containing the common methods for the evaluation of your model.
    Your own submission should NOT modify this script, but only provide your own model when running the
    evaluation.

    Please refer to the README for more details and check submission.py for a sample implementation.

"""
from dataclasses import dataclass
import os
import inspect
import hashlib
import pandas as pd
import time
import numpy as np
import json
from reclist.abstractions import RecList
from evaluation.EvalRSRecList import EvalRSRecList, EvalRSDataset
from collections import defaultdict
from evaluation.utils import download_with_progress, get_cache_directory, LFM_DATASET_PATH, decompress_zipfile, \
    upload_submission, TOP_K_CHALLENGE, LEADERBOARD_TESTS
import requests

class ChallengeDataset:

    def __init__(self, num_folds=4, seed: int = None, force_download: bool = False):
        # download dataset
        self.path_to_dataset = os.path.join(get_cache_directory(), 'evalrs_dataset')
        if not os.path.exists(self.path_to_dataset) or force_download:
            print("Downloading LFM dataset...")
            download_with_progress(LFM_DATASET_PATH, os.path.join(get_cache_directory(), 'evalrs_dataset.zip'))
            decompress_zipfile(os.path.join(get_cache_directory(), 'evalrs_dataset.zip'),
                               get_cache_directory())
        else:
            print("LFM dataset already downloaded. Skipping download.")

        self.path_to_events = os.path.join(self.path_to_dataset, 'evalrs_events.csv')
        self.path_to_tracks = os.path.join(self.path_to_dataset, 'evalrs_tracks.csv')
        self.path_to_users = os.path.join(self.path_to_dataset, 'evalrs_users.csv')

        assert os.path.exists(self.path_to_events)
        assert os.path.exists(self.path_to_tracks)
        assert os.path.exists(self.path_to_users)

        print("Loading dataset.")
        self.df_events = pd.read_csv(self.path_to_events, index_col=0, dtype='int32')
        self.df_tracks = pd.read_csv(self.path_to_tracks,
                                     dtype={
                                         'track_id': 'int32',
                                         'artist_id': 'int32'
                                     }).set_index('track_id')

        self.df_users = pd.read_csv(self.path_to_users,
                                    dtype={
                                        'user_id': 'int32',
                                        'playcount': 'int32',
                                        'country_id': 'int32',
                                        'timestamp': 'int32',
                                        'age': 'int32',
                                    }).set_index('user_id')

        print("Generating folds.")
        self.num_folds = num_folds
        self.unique_user_ids_df = self.df_events[['user_id']].drop_duplicates()
        self._random_state = int(time.time()) if not seed else seed
        self._train_set, self._test_set = self._generate_folds(self.num_folds, self._random_state)

        print("Generating dataset hashes.")
        self._events_hash = hashlib.sha256(pd.util.hash_pandas_object(self.df_events.sample(n=1000,
                                                                                            random_state=0)).values
                                           ).hexdigest()
        self._tracks_hash = hashlib.sha256(pd.util.hash_pandas_object
                                           (self.df_tracks.sample(n=1000, random_state=0)
                                            .explode(['albums', 'albums_id'])).values
                                           ).hexdigest()
        self._users_hash = hashlib.sha256(pd.util.hash_pandas_object(self.df_users.sample(n=1000,
                                                                                          random_state=0)).values
                                          ).hexdigest()


    def _get_vertice_count(self, df, column_name:str):
        df = df.copy()
        df['counts'] = 1
        counts = df.groupby(column_name, as_index=True)[['counts']].sum()
        return counts

    def _k_core_filter(self, df_user_track: pd.DataFrame, k=10):
        num_users_prev, num_tracks_prev = None, None
        delta = True
        iter, max_iter = 0, 10
        valid_users = df_user_track['user_id'].unique()
        valid_tracks = df_user_track['track_id'].unique()
        while delta and iter < max_iter:
            track_counts = self._get_vertice_count(df_user_track, 'track_id')
            valid_tracks = track_counts[track_counts['counts']>=k].index
            # keep only valid tracks
            df_user_track = df_user_track[df_user_track['track_id'].isin(valid_tracks)]
            user_counts = self._get_vertice_count(df_user_track, 'user_id')
            valid_users = user_counts[user_counts['counts']>=k].index
            # keep only valid users
            df_user_track = df_user_track[df_user_track['user_id'].isin(valid_users)]

            num_tracks = len(valid_tracks)
            num_users = len(valid_users)
            # check for any update
            delta = (num_users != num_users_prev) or (num_tracks != num_tracks_prev)

            num_users_prev = num_users
            num_tracks_prev = num_tracks
            iter+=1

            # # DEBUG
            # print("ITER {}".format(iter))
            # print("THERE ARE {} VALID TRACKS; MIN VERTICES {}".format(len(valid_tracks), track_counts['counts'].min()))
            # print("THERE ARE {} VALID USERS; MIN VERTICES {}".format(len(valid_users), user_counts['counts'].min()))

        return valid_users, valid_tracks

    def _generate_folds(self, num_folds: int, seed: int, frac=0.25) -> (pd.DataFrame, pd.DataFrame):

        fold_ids = [(self.unique_user_ids_df.sample(frac=frac, random_state=seed+_)
                     .reset_index(drop=True)
                     .rename({'user_id': _}, axis=1)) for _ in range(num_folds)]
        # in theory all users should have at least 10 interactions
        df_fold_user_ids = pd.concat(fold_ids, axis=1)

        test_dfs = []
        train_dfs_idx = []
        for fold in range(num_folds):
            df_fold_events = self.df_events[self.df_events['user_id'].isin(df_fold_user_ids[fold])]
            # perform k-core filter; threshold of 10
            valid_user_ids, valid_track_ids = self._k_core_filter(df_fold_events[['user_id','track_id']], k=10)

            df_fold_events = df_fold_events[df_fold_events['user_id'].isin(valid_user_ids)]
            df_fold_events = df_fold_events[df_fold_events['track_id'].isin(valid_track_ids)]

            df_groupby = df_fold_events.groupby(by='user_id', as_index=False)
            df_test = df_groupby.sample(n=1, random_state=seed)[['user_id', 'track_id']]
            df_test['fold'] = fold
            df_train = df_fold_events.index.difference(df_test.index).to_frame(name='index')
            df_train['fold'] = fold

            test_dfs.append(df_test)
            train_dfs_idx.append(df_train)
            # unique_user_id = pd.DataFrame(df_fold_events['user_id'].unique(), columns=['user_id'])
            # unique_user_id['fold'] = fold
            # fold_user_ids.append(unique_user_id)

        df_test = pd.concat(test_dfs, axis=0)
        df_train = pd.concat(train_dfs_idx, axis=0)

        # print(df_test)
        # print('====')
        # print(df_users)
        # print('====')
        return df_train, df_test

    def _get_train_set(self, fold: int) -> pd.DataFrame:
        assert fold <= self._test_set['fold'].max()
        train_index =self._train_set[self._train_set['fold']==fold]['index']
        # test_index = self._test_set[self._test_set['fold']==fold].index
        # fold_users = self._fold_ids[self._fold_ids['fold']==fold]['user_id']
        # train_fold = (self.df_events.loc[self.df_events['user_id'].isin(fold_users)])

        return self.df_events.loc[train_index]

    def _get_test_set(self, fold: int, limit: int = None, seed: int =0) -> pd.DataFrame:
        assert fold <= self._test_set['fold'].max()
        test_set = self._test_set[self._test_set['fold'] == fold][['user_id', 'track_id']]
        if limit:
            return test_set.sample(n=limit, random_state=seed)
        else:
            return test_set

    def get_sample_train_test(self):
        return self._get_train_set(1), self._get_test_set(1)


class EvalRSRunner:

    def __init__(self,
                 dataset: ChallengeDataset,
                 email: str = None,
                 participant_id: str = None,
                 aws_access_key_id: str = None,
                 aws_secret_access_key: str = None,
                 bucket_name: str = None):
        self.email = email
        self.participant_id = participant_id
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.bucket_name = bucket_name
        self._num_folds = None
        self._random_state = None
        self._folds = None
        self.model = None
        self.dataset = dataset
        self.secret = 1214210679

        resp = requests.get(url="https://raw.githubusercontent.com/RecList/evalRS-CIKM-2022/main/secret.json")
        data = resp.json()
        if data["secret"] != self.secret:
            raise Exception("Please pull the latest version of the evalRS-CIKM-2022 repository")

    def _test_model(self, model, fold: int, limit: int = None, custom_RecList: RecList = None) -> str:
        # use default RecList if not specified
        myRecList = custom_RecList if custom_RecList else EvalRSRecList

        test_set_df = self.dataset._get_test_set(fold=fold, limit=limit)
        x_test = test_set_df[['user_id']]
        y_test = test_set_df.set_index('user_id')

        dataset = EvalRSDataset()
        dataset.load(x_train=self.dataset._get_train_set(fold=fold),
                     x_test=x_test,
                     y_test=y_test,
                     users=self.dataset.df_users,
                     items=self.dataset.df_tracks)
        # TODO: we should verify the shape of predictions respects top_k=20
        rlist = myRecList(model=model, dataset=dataset)
        rlist.load_dense_repr(path_to_word_vectors=os.path.join(self.dataset.path_to_dataset,'song2vec.wv'))
        report_path = rlist()
        return report_path

    # simple mean for now
    def _aggregate_scores(self, agg_test_results: dict) -> float:
        """Aggregate scores on individual tests.

        Here, we consider the following tests:
        'HIT_RATE', 'MRR', 'MRED_COUNTRY', 'MRED_USER_ACTIVITY',
        'MRED_TRACK_POPULARITY','MRED_ARTIST_POPULARITY',
        'MRED_GENDER', 'BEING_LESS_WRONG', 'LATENT_DIVERSITY' 

        The function returns:

        - leaderboard_score: phase 2 score. It is computed normalizing each test score as
            the relative performance between a chosen baseline and the best results among
            submissions in Phase 1. Here, we use the official CBOW baseline.
            Note that if the submission does not meet the minimum requirements for HR@100
            and MRR, this score is set to 0.

        - p1_score: score assigned using the logic during phase 1. Use this score just 
            for reference with scores in phase 1. WE WILL NOT USE THIS SCORE IN PHASE 2.
        """
        p1_score = np.mean(list(agg_test_results.values()))
        reference = PhaseOne()
        
        # Check if submission meets minimum reqs
        if agg_test_results["HIT_RATE"] < reference.HR_THRESHOLD:
            return 0.0, p1_score
        
        normalized_scores = dict()
        for test in LEADERBOARD_TESTS:
            normalized_scores[test] = (
                agg_test_results[test] - reference.baseline[test]
            ) / (reference.best[test] - reference.baseline[test])

        # Computing meta-scores
        # Performance
        ms_perf = (normalized_scores["HIT_RATE"] + normalized_scores["MRR"]) / 2
        #Fairness / Slicing
        ms_fair = (
            normalized_scores["MRED_COUNTRY"] +
            normalized_scores["MRED_USER_ACTIVITY"] +
            normalized_scores["MRED_TRACK_POPULARITY"] +
            normalized_scores["MRED_ARTIST_POPULARITY"] +
            normalized_scores["MRED_GENDER"]
        ) / 5
        # Behavioral
        ms_behav = (
            normalized_scores["BEING_LESS_WRONG"] + normalized_scores["LATENT_DIVERSITY"]
        ) / 2

        # Meta-scores weights
        w = 1, 1.5, 1.5
        leaderboard_score = (w[0] * ms_perf + w[1] * ms_fair + w[2] * ms_behav) / sum(w)
        
        return leaderboard_score, p1_score

    def evaluate(
            self,
            model,
            seed: int = None,
            upload: bool = True,
            limit: int = 0,
            custom_RecList: RecList = None,
            debug=True,
            # these are additional arguments for training the model, if you need
            # to pass additional stuff
            **kwargs 
    ):
        if debug:
            print('\nBegin Evaluation... ')
        self._random_state = int(time.time()) if not seed else seed
        self.model = model

        self._events_hash = self.dataset._events_hash
        self._users_hash = self.dataset._users_hash
        self._tracks_hash = self.dataset._tracks_hash

        num_folds = self.dataset._test_set['fold'].max() + 1

        if num_folds != 4 or limit != 0:
            print("\nWARNING: default values are not used - upload is disabled")
            upload = False
        # if upload, check we have the necessary credentials
        if upload:
            assert self.email
            assert self.participant_id
            assert self.aws_access_key_id
            assert self.aws_secret_access_key
            assert self.bucket_name

        fold_results_path = []
        # perform training and evaluation for each fold
        # results are automatically stored by reclist to a local path
        # store local path into list
        for fold in range(num_folds):
            train_df = self.dataset._get_train_set(fold=fold)
            if debug:
                print('\nPerforming Training for fold {}/{}...'.format(fold + 1, num_folds))
            self.model.train(train_df, **kwargs)
            if debug:
                print('Performing Evaluation for fold {}/{}...'.format(fold + 1, num_folds))
            results_path = self._test_model(self.model, fold, limit=limit, custom_RecList=custom_RecList)

            fold_results_path.append(results_path)

        raw_results = []
        fold_results = defaultdict(list)
        for fold, results_path in enumerate(fold_results_path):
            with open(os.path.join(results_path, 'results', 'report.json')) as f:
                result = json.load(f)
            # save raw reclist output
            raw_results.append(result)
            # extract tests results which we care about
            for test_data in result['data']:
                if test_data['test_name'] in LEADERBOARD_TESTS:
                    if isinstance(test_data['test_result'], dict) and 'mred' in test_data['test_result']:
                        fold_results[test_data['test_name']].append(test_data['test_result']['mred'])
                    else:
                        fold_results[test_data['test_name']].append(test_data['test_result'])
        print(json.dumps(fold_results, indent=2))
        # compute means for each test over fold
        agg_results = {test: np.mean(res) for test, res in fold_results.items()}
        # generate single score
        leaderboard_score, p1_score = self._aggregate_scores(agg_results)
        print("LEADERBOARD SCORE : {}".format(leaderboard_score))

        # TODO: CI computation ?
        # build final output dict
        out_dict = {
            'reclist_reports': raw_results,
            'results': agg_results,
            'hash': hash(self),
            'secret': self.secret,
            'phase_one_score': p1_score,
            'SCORE': leaderboard_score
        }

        if self.email:
            local_file = '{}_{}.json'.format(self.email.replace('@', '_'), int(time.time() * 10000))
            with open(local_file, 'w') as outfile:
                json.dump(out_dict, outfile, indent=2)
            print('SUBMISSION RESULTS SAVED TO {}'.format(local_file))
            if upload:
                upload_submission(local_file,
                                  aws_access_key_id=self.aws_access_key_id,
                                  aws_secret_access_key=self.aws_secret_access_key,
                                  participant_id=self.participant_id,
                                  bucket_name=self.bucket_name)

    def __hash__(self):
        hash_inputs = [
            self._num_folds,
            self._events_hash,
            self._users_hash,
            self._tracks_hash,
            inspect.getsource(self.evaluate).lstrip(' ').rstrip(' '),
            inspect.getsource(self._test_model).lstrip(' ').rstrip(' '),
            inspect.getsource(self.dataset._get_test_set).lstrip(' ').rstrip(' '),
            inspect.getsource(self.dataset._get_train_set).lstrip(' ').rstrip(' '),
            inspect.getsource(self.dataset._generate_folds).lstrip(' ').rstrip(' '),
            inspect.getsource(self.__init__).lstrip(' ').rstrip(' '),
            inspect.getsource(self.__hash__).lstrip(' ').rstrip(' '),
        ]
        hash_input = '_'.join([str(_) for _ in hash_inputs])
        return int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)


@dataclass
class PhaseOne:

    @property
    def baseline(self):
        return self._CBOW_SCORES

    @property
    def best(self):
        return self._BEST_SCORE_P1

    HR_THRESHOLD = 0.015  # ~ 20% below CBOW HIT RATE
    MRR_THRESHOLD = 1e-5

    _CBOW_SCORES = {
        "HIT_RATE": 0.018763,
        "MRR": 0.001654,
        "MRED_COUNTRY": -0.006944,
        "MRED_USER_ACTIVITY": -0.012460,
        "MRED_TRACK_POPULARITY": -0.006816,
        "MRED_ARTIST_POPULARITY": -0.003915,
        "MRED_GENDER": -0.004354,
        "BEING_LESS_WRONG": 0.2744871, # Original score (0.322926) decreased by 15%
        "LATENT_DIVERSITY": -0.324706
    }

    _BEST_SCORE_P1 = {
        "HIT_RATE": 0.264642,
        "MRR": 0.067493,
        "MRED_COUNTRY": -0.000037,
        "MRED_USER_ACTIVITY": -0.000051,
        "MRED_TRACK_POPULARITY": -0.000036,
        "MRED_ARTIST_POPULARITY": -0.000028,
        "MRED_GENDER": -0.000032,
        "BEING_LESS_WRONG": 0.40635,
        "LATENT_DIVERSITY": -0.202812
    }
