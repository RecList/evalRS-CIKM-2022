"""

Example of a custom implementation for the EvalRSRunner class. In this case, we
override the training method to implement a simple random model.

"""

from evaluation.EvalRSRunner import EvalRSRunner
from models.RandomModel import RandomModel
import pandas as pd


class MyEvalRSRunner(EvalRSRunner):
    def train_model(self, train_df: pd.DataFrame):
        return RandomModel(self.df_tracks)
