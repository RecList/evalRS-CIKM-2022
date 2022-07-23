import numpy as np
import pandas as pd
from reclist.abstractions import RecModel
from typing import List, Union


class RandomModel(RecModel):
    def __init__(self, items: pd.DataFrame):
        super(RandomModel, self).__init__()
        self.items = items

    def predict(self, user_ids: Union[List, np.ndarray], k=10) -> pd.DataFrame:
        if isinstance(user_ids, list):
            user_ids = np.array(user_ids)
        num_users = len(user_ids)
        pred = self.items.sample(n=k*num_users, replace=True)['track_id'].values
        pred = pred.reshape(num_users, k)
        pred = np.concatenate((np.array(user_ids).reshape(-1, 1), pred), axis=1)
        return pd.DataFrame(pred, columns=['user_id', *[str(i) for i in range(k)]]).set_index('user_id')
