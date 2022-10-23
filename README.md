# EvalRS-CIKM-2022
Official Repository for EvalRS @ CIKM 2022: a Rounded Evaluation of Recommender Systems

<a href="https://www.kaggle.com/code/vinidd/cikm-data-challenge"><img src="https://img.shields.io/badge/Kaggle-Notebook-%2355555)"></a> 
<a href="https://colab.research.google.com/drive/1w1fbfCwQKMQNCLbbEF-Qxyin3wE7052T?usp=sharing"> <img src="https://colab.research.google.com/assets/colab-badge.svg"> </a>
<a href="https://medium.com/p/b9fa101ef79a"> <img src="https://raw.githubusercontent.com/aleen42/badges/master/src/medium.svg"> </a>


![https://reclist.io/cikm2022-cup/](images/back_evalrs.png)

*Note*: EvalRS 2022 was held during CIKM 2022 (October 2022) as a data challenge and workshop. This README and the related links provide an overview of the competition, archive workshop artifacts as well as the aftermath of the event. If you are interested in running the evaluation loop _exactly as it was during EvalRS 2022_, the original README, with rules, instructions and full guidelines can be found untouched [here](/README_CIKM_2022.md).

*Important*: while this README is an archive for the event and the workshop, _all the code, data, tests and evaluation methodology_ are still fully available in this very [repository](/README_CIKM_2022.md). If you are working on the evaluation of RecSys, or you wish to run your latest model on Last.FM through a set of diverse tests, you can (and should!) re-use this repository and our leaderboard (as a ready-to-go baseline).

## Overview

This is the official repository for [EvalRS @ CIKM 2022](https://reclist.io/cikm2022-cup/): _a Rounded Evaluation of Recommender Systems_. The aim of the challenge was to evaluate recommender systems across a set of important dimensions (accuracy being _one_ of them) through a principled and re-usable sets of abstractions, as provided by [RecList](https://github.com/jacopotagliabue/reclist) 🚀. EvalRS is based on the [LFM-1b Dataset, Corpus of Music Listening Events for Music Recommendation](http://www.cp.jku.at/datasets/LFM-1b/): participants were asked to solve a typical user-item scenario and recommend new songs to users.

During CIKM 2022, we organized a popular workshop on rounded evaluation for RecSys, including our [reflections](/workshop_slides/challenge_presentation_tagliabue.pdf) as organizers of the event, the best paper [presentation](/workshop_slides/EvalRS_teamML.pdf) and keynotes from renown practitioners in the field, Prof. [Jannach](/workshop_slides/keynote_jannach.pdf) and Prof. [Ekstrand](/workshop_slides/keynote_ekstrand.pdf).

If you are interested in running the same evaluation loop on your own model, re-use our baselines, or simply revisit the rules and guidelines of the original event, please check the official [competition README](/README_CIKM_2022.md). The original README includes also in-depth dataset analysis and explanations on how to run a custom model and add a custom test to [RecList](https://github.com/jacopotagliabue/reclist). For an introduction to the main themes of this competition and details on our methodological choices please refer to the workshop [presentation](/workshop_slides/challenge_presentation_tagliabue.pdf) and [paper](https://arxiv.org/abs/2207.05772).

_Papers, code, presentations from EvalRS are all freely available for the community through this repository: check the appropriate sections below for the Award recipients and the materials provided by organizers and partecipants._

### Quick links

* 🛖 [EvalRS website](https://reclist.io/cikm2022-cup/)
* 📚 [EvalRS paper](https://arxiv.org/abs/2207.05772)
* 📖 [RecList website](https://reclist.io/)

## Organizers

This Data Challenge was built in the open, with the goal of adding lasting artifacts to the community. _EvalRS_ was a collaboration between practitioners from industry and academia, who joined forces to make it happen:

* Jacopo Tagliabue, South Park Commons / NYU
* Federico Bianchi, Stanford
* Tobias Schnabel, Microsoft
* Giuseppe Attanasio, Bocconi University
* Ciro Greco, South Park Commons
* Gabriel de Souza P. Moreira, NVIDIA
* Patrick John Chia, Coveo

For inquiries, please reach out to the corresponding [author](https://www.linkedin.com/in/jacopotagliabue/).

## Sponsors

This Data Challenge was possible thanks to the generous support of these awesome folks. Make sure to add a star to [our library](https://github.com/jacopotagliabue/reclist) and check them out!

<a href="https://neptune.ai/" target="_blank">
    <img src="https://github.com/jacopotagliabue/reclist/raw/main/images/neptune.png" width="200"/>
</a>

<a href="https://www.comet.com/?utm_source=jacopot&utm_medium=referral&utm_campaign=online_jacopot_2022&utm_content=github_reclist" target="_blank">
    <img src="https://github.com/jacopotagliabue/reclist/raw/main/images/comet.png" width="200"/>
</a>

<a href="https://gantry.io/" target="_blank">
    <img src="https://github.com/jacopotagliabue/reclist/raw/main/images/gantry.png" width="200"/>
</a>

## How to Cite

If you find our code, datasets, tests useful in your work, please cite the original WebConf contribution as well as the EvalRS paper.

_RecList_

```
@inproceedings{10.1145/3487553.3524215,
    author = {Chia, Patrick John and Tagliabue, Jacopo and Bianchi, Federico and He, Chloe and Ko, Brian},
    title = {Beyond NDCG: Behavioral Testing of Recommender Systems with RecList},
    year = {2022},
    isbn = {9781450391306},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3487553.3524215},
    doi = {10.1145/3487553.3524215},
    pages = {99–104},
    numpages = {6},
    keywords = {recommender systems, open source, behavioral testing},
    location = {Virtual Event, Lyon, France},
    series = {WWW '22 Companion}
}
```

_EvalRS_

```
@misc{https://doi.org/10.48550/arxiv.2207.05772,
  doi = {10.48550/ARXIV.2207.05772},
  url = {https://arxiv.org/abs/2207.05772},
  author = {Tagliabue, Jacopo and Bianchi, Federico and Schnabel, Tobias and Attanasio, Giuseppe and Greco, Ciro and Moreira, Gabriel de Souza P. and Chia, Patrick John},
  title = {EvalRS: a Rounded Evaluation of Recommender Systems},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Awards

### Student Awards

* Wei-Wei Du
* Flavio Giobergia
* Wei-Yao Wang
* Jinhyeok Park
* Dain Kim

### Best Paper Award

* _Item-based Variational Auto-encoder for Fair Music Recommendation_, by Jinhyeok Park, Dain Kim and Dongwoo Kim (500 USD)

### Best Test Award

* _Variance Agreement_, by Flavio Giobergia (500 USD)

### Leaderboard Awards

Ranking | Team | Score
--- | --- | ---
1 | lik | 1.70
2 | ML | 1.55
3 | fgiobergia | 1.33
4 | wwweiwei | 1.18
5 | Sunshine | 1.14

* First prize, _lyk team_ (3000 USD)
* Second prize, _ML team_ (1000 USD)

## Workshop Presentations

* [EvalRS Challenge Presentation](/workshop_slides/challenge_presentation_tagliabue.pdf)
* [Keynote by Prof. Jannach](/workshop_slides/keynote_jannach.pdf)
* [Keynote by Prof. Ekstrand](/workshop_slides/keynote_ekstrand.pdf)
* [Best paper presentation](/workshop_slides/EvalRS_teamML.pdf)

## Papers and Repositories

* Team wwweiwei, _Track2Vec: Fairness Music Recommendation with a GPU-Free Customizable-Driven Framework_,[code](https://github.com/wwweiwei/Track2Vec) [paper](/final_papers/EvalRS2022_paper_582.pdf)
* Team fgiobergia, _Triplet Losses-based Matrix Factorization for Robust Recommendations_, [code](https://github.com/fgiobergia/CIKM-evalRS-2022/) [paper](/final_papers/EvalRS2022_paper_8348.pdf)
* Team ML, _Item-based Variational Auto-encoder for Fair Music Recommendation_,[code](https://github.com/ParkJinHyeock/evalRS-submission) [paper](/final_papers/EvalRS2022_paper_5248.pdf)
* Team Scrolls, _Bias Mitigation in Recommender Systems to Improve Diversity_,[code](https://github.com/fidelity/jurity/tree/evalrs/evalrs) [paper](/final_papers/EvalRS2022_paper_1487.pdf)
* Team yao0510, _RecFormer: Personalized Temporal-Aware Transformer for Fair Music Recommendation_,[code](https://github.com/wywyWang/RecFormer) [paper](/final_papers/EvalRS2022_paper_4951.pdf)
* Team lyk, _Diversity enhancement for Collaborative Filtering Recommendation_,[code](https://github.com/lazy2panda/evalrs2022_solution) [paper](/final_papers/EvalRS2022_paper_5875.pdf)