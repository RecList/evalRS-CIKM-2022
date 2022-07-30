"""

Example of a custom implementatin for the EvalRSRunner class. In this case, we 
override the training method to implement a simple random model.

"""

from evaluation.EvalRSRunner import EvalRSRunner
from submission.RandomModel import RandomModel
import pandas as pd


class MyEvalRSRunner(EvalRSRunner):

    def train_model(self, train_df: pd.DataFrame, **kwargs):
        """
        Implement here your training logic. Since our example method is a simple random model,
        we actually don't use any training data to build the model, but you should ;-)

        At the end of training, you should return a model class that implements the `predict` method,
        as RandomModel does.
        """
        # kwargs may contain additional arguments in case, for example, you 
        # have data augmentation strategies
        print("Received additional arguments: {}".format(kwargs))
        # return a Model object
        return RandomModel(self.df_tracks, top_k=20)