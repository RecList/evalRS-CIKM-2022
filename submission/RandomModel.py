import numpy as np
import pandas as pd
from reclist.abstractions import RecModel


class RandomModel(RecModel):
    
    def __init__(self, items: pd.DataFrame, top_k: int=20):
        super(RandomModel, self).__init__()
        self.items = items
        self.top_k = top_k

    def train(self, train_df: pd.DataFrame, **kwargs):
        """
        Implement here your training logic. Since our example method is a simple random model,
        we actually don't use any training data to build the model, but you should ;-)

        At the end of training, you should return a model class that implements the `predict` method,
        as RandomModel does.
        """
        # kwargs may contain additional arguments in case, for example, you
        # have data augmentation strategies
        print("Received additional arguments: {}".format(kwargs))

        print(train_df.head(1))

    def predict(self, user_ids: pd.DataFrame) -> pd.DataFrame:
        """
        
        This function takes as input all the users that we want to predict the top-k items for, and 
        returns all the predicted songs.

        While in this example is just a random generator, the same logic in your implementation 
        would allow for batch predictions of all the target data points.
        
        """
        k = self.top_k
        num_users = len(user_ids)
        pred = self.items.sample(n=k*num_users, replace=True).index.values
        pred = pred.reshape(num_users, k)
        pred = np.concatenate((user_ids[['user_id']].values, pred), axis=1)
        return pd.DataFrame(pred, columns=['user_id', *[str(i) for i in range(k)]]).set_index('user_id')
