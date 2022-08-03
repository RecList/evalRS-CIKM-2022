"""

    This script contains the list of tests that has been prepared for the challenge. Tests are standard
    IR metrics, interesting datasets slices based on data analysis and literature, and novel behavioral tests.

    While you should not modify this class, your submission should contain a class extending "EvalRSRecList"
    with custom tests that you found useful to debug the behavior of your model, or check for 
    regression errors in test cases you care about (check the README and notebooks for more details)

    You're free to experiment with the dataset, and share your insightful work as operationalized in RecTests with
    the community!

"""
import numpy as np
import pandas as pd
from reclist.abstractions import RecList, RecDataset, rec_test


class EvalRSRecList(RecList):


    def mrr_at_k_slice(self,
                        y_preds: pd.DataFrame,
                        y_test: pd.DataFrame,
                        slice_info: pd.DataFrame,
                        slice_key: str,
                        k: int):

        from reclist.metrics.standard_metrics import rr_at_k
        # get rr (reciprocal rank) for each prediction made
        rr = rr_at_k(y_preds, y_test, k=k)
        # convert to DataFrame
        rr = pd.DataFrame(rr, columns=['rr'], index=y_test.index)
        # grab slice info
        rr[slice_key]  =slice_info[slice_key].values
        # group-by slice and get per-slice mrr
        return rr.groupby(slice_key)['rr'].agg('mean').to_json()


    @rec_test('stats')
    def stats(self):
        tracks_per_users = (self._y_test.values!=-1).sum(axis=1)
        return {
            'num_users': len(self._x_test['user_id'].unique()),
            'max_items': int(tracks_per_users.max()),
            'min_items': int(tracks_per_users.min())
        }

    @rec_test('HIT_RATE')
    def hit_rate_at_20(self):
        from reclist.metrics.standard_metrics import hit_rate_at_k
        hr = hit_rate_at_k(self._y_preds, self._y_test, k=20)
        return hr

    @rec_test('MRR')
    def mrr_at_20(self):
        from reclist.metrics.standard_metrics import mrr_at_k
        return mrr_at_k(self._y_preds, self._y_test, k=20)

    @rec_test('MRR_COUNTRY')
    def mrr_at_20_country(self):
        user_countries = self.product_data['users'].loc[self._y_test.index, ['country']]
        return self.mrr_at_k_slice(self._y_preds,
                                   self._y_test,
                                   user_countries,
                                   'country',
                                   k=20)

    @rec_test('MRR_ACTIVITY')
    def mrr_at_20_activity(self):
        bins = np.array([10,100,1000,10000])
        user_activity = self._x_train[self._x_train['user_id'].isin(self._y_test.index)]
        user_activity = user_activity.groupby('user_id',as_index=True, sort=False)[['user_track_count']].sum()
        user_activity = user_activity.loc[self._y_test.index]

        user_activity['bin_index'] = np.digitize(user_activity.values.reshape(-1), bins)
        user_activity['bins'] = bins[user_activity['bin_index'].values-1]

        return self.mrr_at_k_slice(self._y_preds,
                                   self._y_test,
                                   user_activity,
                                   'bins',
                                   k=20)


class EvalRSDataset(RecDataset):
    def load(self, **kwargs):
        self._x_train = kwargs.get('x_train')
        self._x_test = kwargs.get('x_test')
        self._y_test = kwargs.get('y_test')
        self._catalog = {
            'users': kwargs.get('users'),
            'items': kwargs.get('items')
        }
