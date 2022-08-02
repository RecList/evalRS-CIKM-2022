"""

    This script contains the list of tests that has been prepared for the challenge. Tests are standard
    IR metrics, interesting datasets slices based on data analysis and literature, and novel behavioral tests.

    While you should not modify this class, your submission should contain a class extending "EvalRSRecList"
    with custom tests that you found useful to debug the behavior of your model, or check for 
    regression errors in test cases you care about (check the README and notebooks for more details)

    You're free to experiment with the dataset, and share your insightful work as operationalized in RecTests with
    the community!

"""
import pandas as pd
from reclist.abstractions import RecList, RecDataset, rec_test


class EvalRSRecList(RecList):

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
        from reclist.metrics.standard_metrics import rr_at_k
        # get reciprocal rank for each prediction made
        rr = rr_at_k(self._y_preds, self._y_test, k=20)
        # convert to DataFrame
        rr = pd.DataFrame(rr, columns=['rr'])
        rr.index = self._y_preds.index
        # get country info
        rr['country'] = self.product_data['users'].loc[self._x_test['user_id'], 'country'].values
        # group-by slice and get per-slice mrr
        return rr.groupby('country')['rr'].agg('mean').to_json()

class EvalRSDataset(RecDataset):
    def load(self, **kwargs):
        self._x_test = kwargs.get('x_test')
        self._y_test = kwargs.get('y_test')
        self._catalog = {
            'users': kwargs.get('users'),
            'items': kwargs.get('items')
        }
