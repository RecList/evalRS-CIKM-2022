
import os
import pandas as pd
from dotenv import load_dotenv
from evaluation.EvalRSRunner import EvalRSRunner
from models.RandomModel import RandomModel

load_dotenv('../upload.env')

EMAIL = os.getenv('EMAIL')  # the e-mail you used to sign up
BUCKET_NAME = os.getenv('BUCKET_NAME')  # you received it in your e-mail
PARTICIPANT_ID = os.getenv('PARTICIPANT_ID')  # you received it in your e-mail
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')  # you received it in your e-mail
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')  # you received it in your e-mail


class RandomEvalRSRunner(EvalRSRunner):
    def train_model(self, train_df: pd.DataFrame):
        return RandomModel(self.df_tracks)


if __name__ == '__main__':

    runner = RandomEvalRSRunner(path_to_dataset='../lfm_1b_dataset',
                                num_folds=4,
                                aws_access_key_id=AWS_ACCESS_KEY,
                                aws_secret_access_key=AWS_SECRET_KEY,
                                participant_id=PARTICIPANT_ID,
                                bucket_name=BUCKET_NAME,
                                email=EMAIL)
    runner.evaluate(upload=True, limit=1000000)
