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
load_dotenv('upload.env', verbose=True)


EMAIL = os.getenv('EMAIL')  # the e-mail you used to sign up
# if you're testing this code locally, you can use a dummy e-mail address
# and set UPLOAD to 0 in the env file, and the code will not upload the results
assert EMAIL != '' and EMAIL is not None
BUCKET_NAME = os.getenv('BUCKET_NAME')  # you received it in your e-mail
PARTICIPANT_ID = os.getenv('PARTICIPANT_ID')  # you received it in your e-mail
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')  # you received it in your e-mail
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')  # you received it in your e-mail
UPLOAD = bool(os.getenv('UPLOAD'))  # it's a boolean, True if you want to upload your submission
LIMIT = int(os.getenv('LIMIT'))  # limit the number of test cases, for quick tests / iterations. 0 for no limit

print("Submission will be uploaded: {}".format(UPLOAD))
if LIMIT > 0:
    print("WARNING: only {} test cases will be used".format(LIMIT))

# run the evaluation loop when the script is called directly
if __name__ == '__main__':
    # import YOUR runner class, which is an instance of the general EvalRSRunner class
    from submission.my_runner import MyEvalRSRunner
    print('\n\n==== Starting evaluation script at: {} ====\n'.format(datetime.utcnow()))
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
    runner.evaluate(upload=UPLOAD, limit=LIMIT)
    print('\n\n==== Evaluation ended at: {} ===='.format(datetime.utcnow()))
