# EvalRS-CIKM-2022 : Evaluation Loop

`EvalRSRunner` (defined in`EvalRSRunner.py`) is an abstract class which encapsulates the evaluation approach for EvalRS (Bootstrapped Nested Cross-Validation). Being this challenge a _code competiton_ on a public dataset, we could not rely on unseen test data to produce a final leaderboard.

[comment]: <> (Describe BNCV in some detail)


## How to use the provided abstractions

### Main abstractions

We provide two basic abstractions for the Data Challenge:

* an `EvalRSRunner` class to get the dataset, run the evaluation and submit the scores to the board. _You should not change this class in any way_;
* a `EvalRSRecList` class, implementing a RecList class from the [OS package](https://reclist.io/). _You should not change this class in any way, but you are encouraged to i) understand it, ii) extend it with your own tests for your paper (see the `notebooks` folder for a working example)_. See below for an explanation of the provided tests.

We also provide out-of-the-box utility functions and a template script as an entry point for your final submission (`submission.py`). As long as your training and prediction logic respects the provided API, you should be able to easily adapt your model to the Data Challenge.

During the _leaderboard phase_, you can submit your scores to the leaderboard running the code however you prefer. However, at _submission_, your repository needs to comply with the rules in the general `README`. Remember: if we are not able to reproduce your results and statistically verify your scores, you won't be eligible for the prize.

### Build your own evaluation loop

To make a new submission to the leaderboard you are just required to build two new objects:

* an implementation of `EvalRSRunner` containing your training logic;
* a model object built inside the `train_model` method of your custom runner.

First, you should inherit `EvalRSRunner` and implement the abstract method `train_model`: `train_model` should contain the model training code, _including any necessary hyper-parameter optimization_. By inheriting `EvalRSRunner`, you get access to necessary meta-data via `self.df_tracks` and `self.df_users`: you are free to use any modelling technique you want (collaborative filtering, two-tower etc.) as long as your code complies with the Data Challenge rules (no test leaking, hyperparameter and compute time within the budget etc.).

Second, when the training is done, you should wrap your model in an object with a method `predict`: `train_model` should return this object as a result of training, since this is what the evaluation loop will use to get predictions and score them. The `predict` method accepts as input a dataframe of all the user IDs for which the model is asked to make a prediction on.

For each `user_id`, we expect `k` predictions (where `k=20`): you can play around with different Ks for debugging purposes _but_ only `k=20` will be accepted for the leaderboard. [The expected prediction output is a dataframe](../images/prediction.jpg) with `user_id` as index and k columns, each representing the ranked recommendations (0th column being the highest rank). In addition, it is expected that the predictions are in the same order as the `user_id` in the input dataframe. An example of the desired dataframe format for `n` `user_ids` and `k` predictions per user is seen in the table below. Note that if your model provides less than `k` predictions for a given `user_id`, 
the empty columns should be filled with `-1`. 

 |           |  0          | ...        | k-1         | 
| ---------- | ----------  | ---------- | ----------- |
| user_id_1  | track_id_1  | ...        | -1          |
| user_id_2  | track_id_4  | ...        | track_id_5  |
| ...        | ...         | ...        | -1          |
| user_id_n  | track_id_18 | ...        | track_id_9  |


_Implementing the class, and returning the trained model in the proper wrapper:_

```python

class MyEvalRSRunner(EvalRSRunner):
    def train_model(self, train_df: pd.DataFrame):
        """
        Inherit from the Challenge class EvalRSRunner, and implement your training logic
        in this function. Return a trained model.
        """
        # do your magic here
        model.train(train_df)
        # store your model into a model object - see below
        my_model = MyModel(model, top_k=20)
        # return the trained model in the proper wrapper
        return my_model
```

_Example of a model wrapper implementing the predict method_

```python

class MyModel(RecModel):
    def __init__(self, model, top_k=20):
       # implement your init logic here
       self._model = model
       self.top_k = top_k

    def predict(self, user_ids: pd.DataFrame) -> pd.DataFrame:
        """
        Implement your logic here: given the user Ids in the test set, recommend the top-k songs for them.
        """
        k = self.top_k
        # do your magic here
        pred = self._model.predict(user_ids, k)
        
        return pd.DataFrame(pred, columns=['user_id', *[ str(i) for i in range(k)]]).set_index('user_id')

```

As your final code submission, you are also required to contribute a new test: make sure to include an extended RecList in your code.

Please see the `notebooks` folder for a walk-through on the evaluation engine, and check the instructions in the main `README` and the template script in `submission.py` to understand how to make the final code submission.

## Last.FM RecList

_TBC_
