"""

    Template script for the submission. You can use this as a starting point for your code: you can
    copy this script as is into your repository, and then modify the associated Runner class to include
    your logic, instead of the random baseline.

    Please make sure you read and understand the competition rule and guidelines before you start.

"""

import os
from datetime import datetime
from dotenv import load_dotenv

# import env variables from file
load_dotenv('../upload.env', verbose=True)


EMAIL = os.getenv('EMAIL')  # the e-mail you used to sign up
BUCKET_NAME = os.getenv('BUCKET_NAME')  # you received it in your e-mail
PARTICIPANT_ID = os.getenv('PARTICIPANT_ID')  # you received it in your e-mail
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')  # you received it in your e-mail
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')  # you received it in your e-mail


# run the evaluation loop when the script is called directly
if __name__ == '__main__':
    # import YOUR runner class, which is an instance of the general EvalRSRunner class
    from submission.my_runner import MyEvalRSRunner
    print('==== Starting evaluation script at: {} ====\n'.format(datetime.utcnow()))
    # run the evaluation loop
    runner = MyEvalRSRunner(
        num_folds=4,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        participant_id=PARTICIPANT_ID,
        bucket_name=BUCKET_NAME,
        email=EMAIL
        )
    print('==== Runner loaded, starting loop at: {} ====\n'.format(datetime.utcnow()))
    runner.evaluate(upload=True, limit=1000000)
    print('\n\n==== Evaluation ended at: {} ===='.format(datetime.utcnow()))
