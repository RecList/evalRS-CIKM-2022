# EvalRS-CIKM-2022
Official Repository for EvalRS @ CIKM 2022: a Rounded Evaluation of Recommender Systems

## Overview

*IMPORTANT*: [EvalRS](https://reclist.io/cikm2022-cup/) is a Data Challenge at [CIKM 2022](https://www.cikm2022.org/). This in-progress repository will host the official scripts and rules for the competition, which is planned for August 2022: add your e-mail [to challenge list](https://docs.google.com/forms/d/e/1FAIpQLSfAypzM1mvd79JfRGRbb9QMfXGMoVYosdjU9C4NFEWNSNUZXQ/viewform) to be notified of important events.

This is the official repository for _EvalRS @ CIKM 2022: a Rounded Evaluation of Recommender Systems_. The aim of the challenge is to evaluate recommender systems across a set of important dimensions (accuracy being _one_ of them) through a principled and re-usable sets of abstractions, as provided by [RecList](https://github.com/jacopotagliabue/reclist).

Please refer to the appropriate sections below to know how to register for the challenge and how to run the evaluation loop properly. For questions about the prize, the provided scripts and the rules, please join our Slack (_forthcoming_).

_We are working hard on this: check back often for updates._

### Quick Links

* [EvalRS website](https://reclist.io/cikm2022-cup/)
* EvalRS paper (_forthcoming_)
* [RecList website](https://reclist.io/)

## Submission

### Run the example submission first

A sample submission script is included in this repository as a template. We suggest you run it _as soon as you have received your credentials_ to check that your setup is correct. To do so:

* create and activate a virtual environment in the root, `python -m venv venv`, `source venv/bin/activate`;
* install the required dependencies in `requirements.txt`;
* create (do _not_ commit it!) a `upload.env` file in the root starting from `local.env` as template, and fill it with your credentials;
* run `python submission.py`.

At the end of the run, the code will have downloaded the dataset for you (so, subsequent runs won't do it anymore), ran the random model baseline, computed your score and sent them to the leaderboard: if you refresh the challenge page (LINK), you should see your submission there. Congrats, everything is working! Now, time to win the challenge!

P.s.: if something in the procedure goes wrong, please contact us through Slack!

### Run your code

A valid submission script can be obtained by copying into your repository `submission.py`, and modify `my_runner.py` and the related model to use your logic instead of the default one. In theory, no change should be necessary to `submission.py`. Your submission is required to build an instance of the class `EvalRSRunner`, providing an implementation for the `train_model.py` method.

In the `evaluation` folder, we included a lenghtier explanation of the evaluation code involved in this challenge: please refer to that file and the provided examples for in-depth explanation, or reach out on Slack if you're unsure on how to use it.

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