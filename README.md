# EvalRS-CIKM-2022
Official Repository for EvalRS @ CIKM 2022: a Rounded Evaluation of Recommender Systems

![https://reclist.io/cikm2022-cup/](images/back_evalrs.png)

## Overview

*IMPORTANT*: [EvalRS](https://reclist.io/cikm2022-cup/) is a Data Challenge at [CIKM 2022](https://www.cikm2022.org/). This in-progress repository will host the official scripts and rules for the competition, which is planned for August 2022: add your e-mail [to challenge list](https://docs.google.com/forms/d/e/1FAIpQLSfAypzM1mvd79JfRGRbb9QMfXGMoVYosdjU9C4NFEWNSNUZXQ/viewform) to be notified of important events.

This is the official repository for _EvalRS @ CIKM 2022: a Rounded Evaluation of Recommender Systems_. The aim of the challenge is to evaluate recommender systems across a set of important dimensions (accuracy being _one_ of them) through a principled and re-usable sets of abstractions, as provided by [RecList](https://github.com/jacopotagliabue/reclist) 🚀.

Please refer to the appropriate sections below to know how to register for the challenge and how to run the evaluation loop properly. For questions about the prize, the provided scripts and the rules, please join our [Slack](https://reclist.io/cikm2022-cup/).

_We are working hard on this: check back often for updates._

### Quick Links

* 🛖 [EvalRS website](https://reclist.io/cikm2022-cup/)
* 📚 [EvalRS paper](https://arxiv.org/abs/2207.05772)
* 📖 [RecList website](https://reclist.io/)


## Dataset and target scenario

This Data Challenge is based on the [LFM-1b Dataset, Corpus of Music Listening Events for Music Recommendation](http://www.cp.jku.at/datasets/LFM-1b/). The use case is a typical user-item recommendation scenario: at _predction time_, we have a set of target users to which we need to recommend a set of songs to listen to. To achieve that, we have historical anonymous data on previous music consumptions from users in the same setting.

Among all possible datasets, we picked LFM as it suits the spirit and the goal of this Challenge: in particular, thanks to [rich meta-data on users and items](http://www.cp.jku.at/people/schedl/Research/Publications/pdf/schedl_ijmir_2017.pdf), the dataset allows us to test recommender systems among many non-obvious dimensions, on top of standard Information Retrieval Metrics (for the philosophy behind behavioral testing, please refer to the original [RecList paper](https://arxiv.org/abs/2111.09963)).

### Data overview

When you run the evaluation loop below (below), the code will automatically download _a chosen subset of the LFM dataset_, ready to be used (the code will download it only the first time you run it). There are three main objects available from the provided evaluation class:

_Users_: a collection of users and available meta-data, including patterns of consumption, demographics etc.. In the Data Challenge scenario, the user Id is the query item for the model, which is asked to recommend songs to the user.

![http://www.cp.jku.at/datasets/LFM-1b/](images/users.png)

_Tracks_: a collection of tracks and available meta-data. In the Data Challenge scenario, tracks are the target items for the model, i.e. the collection to chose from when the model needs to provide recommendations.

![http://www.cp.jku.at/datasets/LFM-1b/](images/tracks.png)


_Historical Interactions_: a collection of interactions between users and tracks, that is, listening events, which should be used by your model to build the recommender system for the Data Challenge.

![http://www.cp.jku.at/datasets/LFM-1b/](images/training.png)


For in-depth explanations on the code and the template scripts, see the instructions below and check the provided examples and tutorials in `notebooks` and `baselines`.

For information on how the original dataset was built and what meta-data are available, please refer to [this paper](http://www.cp.jku.at/people/schedl/Research/Publications/pdf/schedl_ijmir_2017.pdf).


## How To Join The Race

### First Step: Register

* Register [Online](https://reclist.io/cikm2022-cup/leaderboard.html). You should get some tokens, save them for later! 

### Second Step: Run The Example

A sample submission script is included in this repository as a template. We suggest you run it _as soon as you have received your credentials_ to check that your setup is correct. To do so:

A couple of commands to start; downlad the repo and setup a virtual environment.

```bash
git clone https://github.com/RecList/evalRS-CIKM-2022
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

We use dot files to manage secret keys. Copy the `local.env` file and create an 
`upload.env` (**DO NOT** commit this file). You can fill this file with the keys you got
at step 1.

Now, run

```bash 
python submission.py
```

This first rul will also download the dataset for you (and it won't download it next). The code
runs a random model baseline, computes your score and sent them to the leaderboard. 
Click [here](https://reclist.io/cikm2022-cup/leaderboard.html) to see the leaderboard! It should look like this:

![https://reclist.io/cikm2022-cup/leaderboard.html](images/leaderboard.png)

🎉 🎉 🎉 Congrats, everything is working! Now, time to win the challenge!

P.s.: if something in the procedure goes wrong, please contact us through Slack!

### Third Step: Run your code

A valid submission script can be obtained by copying into your repository `submission.py`, and modify `my_runner.py` and the related model to use your logic instead of the default one. In theory, no change should be necessary to `submission.py`. Your submission is required to build an instance of the class `EvalRSRunner`, providing an implementation for the `train_model.py` method.

In the `evaluation` folder, we included a lenghtier explanation of the evaluation code involved in this challenge; in the `notebooks` folder, we include a step-by-step, heavily commented guide on how to build a submission, including sample data points and an example of using a derived RecList for evaluation; in the `baselines` folder, you will find more complex models than the simple random one, as an inspiration. Remember: this competition as about models as much as about _data_ and _testing_ - take our initial work just as an inspiration!

Please refer to provided examples for in-depth explanations, and don't forget to reach out on Slack if you have any doubt.

## How easy it is to join?

Very easy! If you already have a recommendation model you just need to wrap this in a way that
is consistent with our own API. 

### Training

```python

class MyEvalRSRunner(EvalRSRunner):
    def train_model(self, train_df: pd.DataFrame):
        """
        Inherit from the Challenge class EvalRSRunner, and implement your training logic
        in this function. Return a trained model.
        """
        my_model = MyModel(self.df_tracks)
        my_model.train()
        # return the trained model
        return my_model
```

### Prediction

```python

class MyModel:
    def __init__(self, items: pd.DataFrame):
        self.items = items

    def predict(self, user_ids: pd.DataFrame, k=20) -> pd.DataFrame:
        """
        Implement your logic here: given the user Ids in the test set, 
        recommend the top-k songs for them.
        """
        user_ids = user_ids['user_id'].values
        num_users = len(user_ids)
        pred = self.items.sample(n=k*num_users, replace=True)['track_id'].values
        pred = pred.reshape(num_users, k )
        pred = np.concatenate((np.array(user_ids).reshape(-1,1), pred), axis=1)
        
        return pd.DataFrame(pred, columns=['user_id', *[ str(i) for i in range(k)]]).set_index('user_id')

```

## Data Challenge Structures and Rules

### How the Data Challenge runs

_TBC_

### Rules

_TBC_

### Call for Papers

_TBC_


## FAQ

_TBC_

## Organizers

This Data Challenge focuses on building in the open, and adding lasting artifacts to the community. _EvalRS @ CIKM 2022_ is a collaboration between practitioners from industry and academia, who joined forces to make it happen:

* Jacopo Tagliabue, Coveo / NYU
* Federico Bianchi, Stanford
* Tobias Schnabel, Microsoft
* Giuseppe Attanasio, Bocconi University
* Ciro Greco, Coveo
* Gabriel de Souza P. Moreira, NVIDIA
* Patrick John Chia, Coveo

## Sponsors

TBC

## How to Cite

TBC