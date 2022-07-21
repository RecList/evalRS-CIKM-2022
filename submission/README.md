                                          # EvalRS-CIKM-2022 : Evaluation

`EvalRSRunner` (defined in`submission.py`) is an abstract class which encapsulates the evaluation approach for EvalRS (
Bootstrapped Nested Cross-Validation). 


[comment]: <> (Maybe describe BNCV in some detail)


## How To Use

In order to evaluate and make a submission
to the leaderboard, you should inherit `EvalRSRunner` and implement the abstract method `train_model`. `train_model` 
should contain the model training code, including any necessary hyper-parameter optimization. It should
return a model object (NOTE: potentially a RecModel) with a method `predict` that is used during evaluation. By inheriting
`EvalRSRunner`, you get access to necessary meta-data via `self.df_tracks` and `self.df_users`.

Please see the notebook `random_model.ipynb` for an example of how to use our evaluation engine.

