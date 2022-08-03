# EvalRS-CIKM-2022 : Evaluation Loop

`EvalRSRunner` (defined in`EvalRSRunner.py`) is a class which encapsulates the evaluation approach for EvalRS (Bootstrapped Nested Cross-Validation). Being this challenge a _code competiton_ on a public dataset, we could not rely on unseen test data to produce a final leaderboard.

Our approach is illustrated in the following diagram: subsets of the original dataset are randomly designated as train and test set; `EvalRSRunner` will automatically feed them to the model object you provide (with your training and prediction logic), and the predictions will be scored with RecList. Sampling is per-user, and testing is based on the "leave-one-out" principle: for each user, we pick a track and leave it for the test set as the target item for the predictions.

![Loop explanation](../images/loop.jpg)

This procedure will be repetead _n_ times, and your average scores will be uploaded at the end (the library will take care of it) and determine your position on the leaderboard. Due to the stochastic nature of the loop, scores will vary slightly between runs, but the organizing committee will still be able to statistically evaluate your code submission in the light of your scores. As stated in the rules, since this is a code competition, _reproducibility_ is essential: please take your time to make sure that you understand the submission scripts and that your final project is easily reproducible from scracth.

As demonstrated by the notebook and `submission.py`, you do _not_ have to worry about any of the above implementation details: folds are generated for you and predictions are automatically compared to ground truths by the provided code and RecList.

_Note_: when you instantiate the `ChallengeDataset`, `num_folds` determines how many times the evaluation procedure is run. Default is 4, and that is the value you should use for all your leaderboard submissions.

## How to use the provided abstractions

### Main abstractions

We provide two basic abstractions for the Data Challenge:

* an `EvalRSRunner` class to run the evaluation and submit the scores to the board. _You should not change this class in any way_;
* a `EvalRSRecList` class, implementing a RecList class from the [OS package](https://reclist.io/). _You should not change this class in any way, but you are encouraged to i) understand it, ii) extend it with your own tests for your paper (see the `notebooks` folder for a working example)_. See below for an explanation of the provided tests.

We also provide out-of-the-box utility functions and a template script as an entry point for your final submission (`submission.py`). As long as your training and prediction logic respects the provided model API, you should be able to easily adapt existing models to the Data Challenge.

During the _leaderboard phase_, you can submit your scores to the leaderboard running the code however you prefer. However, at _submission_, your repository needs to comply with the rules in the general `README`. Remember: if we are not able to reproduce your results and statistically verify your scores, you won't be eligible for the prize.

### Build your own evaluation loop

To make a new submission to the leaderboard you are just required to build one new object, a model class containing a `train` and `predict` methods, such as the one contained in the `submission` folder.

First, you should inherit `RecModel` and implement the `train` method: `train` should contain the model training code, _including any necessary hyper-parameter optimization_; if you wish to pass additional parameters to the training function, you can always add them in your init and retrieve them later. You are free to use any modelling technique you want (collaborative filtering, two-tower etc.) as long as your code complies with the Data Challenge rules (no test leaking, hyperparameter and compute time within the budget etc.).

Second, when the training is done, you should wrap your predictions in a method `predict`: `train` should store the trained model inside the class, and `predict` will use that model to provide predictions. The `predict` method accepts as input a dataframe of all the user IDs for which the model is asked to make a prediction on.

For each `user_id`, we expect `k` predictions (where `k=20` for the competition): you can play around with different Ks for debugging purposes _but_ only `k=20` will be accepted for the leaderboard. [The expected prediction output is a dataframe](../images/prediction.jpg) with `user_id` as index and k columns, each representing the ranked recommendations (0th column being the highest rank). In addition, it is expected that the predictions are in the same order as the `user_id` in the input dataframe. An example of the desired dataframe format for `n` `user_ids` and `k` predictions per user is seen in the table below. Note that if your model provides less than `k` predictions for a given `user_id`, 
the empty columns should be filled with `-1`. 

 |           |  0          | ...        | k-1         | 
| ---------- | ----------  | ---------- | ----------- |
| user_id_1  | track_id_1  | ...        | -1          |
| user_id_2  | track_id_4  | ...        | track_id_5  |
| ...        | ...         | ...        | -1          |
| user_id_n  | track_id_18 | ...        | track_id_9  |


_Implementing the class, and returning the trained model in the proper wrapper:_

```python

class MyModel(RecModel):
    
    def __init__(self, items: pd.DataFrame, top_k: int=20, **kwargs):
        super(MyModel, self).__init__()
        self.items = items
        self.top_k = top_k
        # kwargs may contain additional arguments in case, for example, you
        # have data augmentation strategies
        print("Received additional arguments: {}".format(kwargs))
        return

    def train(self, train_df: pd.DataFrame):
        """
        Implement here your training logic. Since our example method is a simple random model,
        we actually don't use any training data to build the model, but you should ;-)

        At the end of training, make sure the class contains a trained model you can use in the predict method.
        """
        print(train_df.head(1))
        print("Training completed!")
        return 

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

```

As your final code submission, you are also required to contribute a new test: make sure to include an extended RecList in your code.

Please see the `notebooks` folder for a walk-through on the evaluation engine, and check the instructions in the main `README` and the template script in `submission.py` to understand how to make the final code submission.

## Last.FM RecList

_TBC_
