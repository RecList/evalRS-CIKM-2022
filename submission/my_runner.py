"""

Example of a custom implementatin for the EvalRSRunner class. In this case, we 
override the training method to implement a simple random model.

"""

from evaluation.EvalRSRunner import EvalRSRunner
from submission.RandomModel import RandomModel
import pandas as pd


class MyEvalRSRunner(EvalRSRunner):
    def train_model(self, train_df: pd.DataFrame):
        """
        Implement here your training logic. Since our example method is a simple random model,
        we actually don't use any training data to build the model, but you should ;-)

        At the end of training, you should return a model class that implements the `predict` method,
        as RandomModel does.
        """
        return RandomModel(self.df_tracks)