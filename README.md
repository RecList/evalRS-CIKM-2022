# EvalRS-CIKM-2022
Official Repository for EvalRS @ CIKM 2022: a Rounded Evaluation of Recommender Systems

<a href="https://www.kaggle.com/code/vinidd/cikm-data-challenge"><img src="https://img.shields.io/badge/Kaggle-Notebook-%2355555)"></a> 
<a href="https://colab.research.google.com/drive/1w1fbfCwQKMQNCLbbEF-Qxyin3wE7052T?usp=sharing"> <img src="https://colab.research.google.com/assets/colab-badge.svg"> </a>
<a href="https://medium.com/p/b9fa101ef79a"> <img src="https://raw.githubusercontent.com/aleen42/badges/master/src/medium.svg"> </a>


![https://reclist.io/cikm2022-cup/](images/back_evalrs.png)

*Note*: EvalRS 2022 was held during CIKM 2022 (October 2022) as a data challenge and workshop. This README and the related links provide an overview of the competition and archive the artifacts from the event to benefit the RecSys community. If you are interested in running the evaluation loop _exactly as it was during EvalRS 2022_, the original README, with rules, instructions and full guidelines can be found untouched [here](/README_CIKM_2022.md).

*Important*: while this README is an archive for the event and the workshop, _all the code, data, tests and evaluation methodology_ are still fully available in this very [repository](/README_CIKM_2022.md). If you are working on the evaluation of RecSys, or you wish to run your latest model on Last.FM through a set of diverse tests, you can (and should!) re-use this repository and our leaderboard (as a ready-to-go baseline).

## Overview

This is the official repository for [EvalRS @ CIKM 2022](https://reclist.io/cikm2022-cup/): _a Rounded Evaluation of Recommender Systems_. The aim of the challenge was to evaluate recommender systems across a set of important dimensions (accuracy being _one_ of them) through a principled and re-usable sets of abstractions, as provided by [RecList](https://github.com/jacopotagliabue/reclist) 🚀. EvalRS is based on the [LFM-1b Dataset, Corpus of Music Listening Events for Music Recommendation](http://www.cp.jku.at/datasets/LFM-1b/): participants were asked to solve a typical user-item scenario and recommend new songs to users.

During CIKM 2022, we organized a popular workshop on rounded evaluation for RecSys, including our [reflections](/workshop_slides/challenge_presentation_tagliabue.pdf) as organizers of the event, the best paper [presentation](/workshop_slides/EvalRS_teamML.pdf) and keynotes from two renown practitioners, Prof. [Jannach](/workshop_slides/keynote_jannach.pdf) and Prof. [Ekstrand](/workshop_slides/keynote_ekstrand.pdf).

If you are interested in running the same evaluation loop on your own model, re-use our baselines, or simply revisit the rules and guidelines of the original event, please check the official [competition README](/README_CIKM_2022.md). The original README includes also in-depth dataset analyses and explanations on how to run a model and add a custom test to [RecList](https://github.com/jacopotagliabue/reclist). For an introduction to the main themes of this competition and details on our methodology, please refer to the workshop [presentation](/workshop_slides/challenge_presentation_tagliabue.pdf) and [paper](https://arxiv.org/abs/2207.05772).

_[Papers](https://ceur-ws.org/Vol-3318/), code, presentations from EvalRS are all freely available for the community through this repository: check the appropriate sections below for the Award recipients and the materials provided by organizers and partecipants._

If you like our work and wish to support open source RecSys projects, please take a second to add a star to [RecList repository](https://github.com/jacopotagliabue/reclist).

### Quick links

* 📖 [RecList website](https://reclist.io/)
* 📖 [Running EvalRS evaluation on your model](https://github.com/RecList/evalRS-CIKM-2022/blob/main/README_CIKM_2022.md)
* 📖 [Our Challenge Review for Nature Machine Intelligence](https://www.nature.com/articles/s42256-022-00606-0)
* 🛖 [EvalRS YouTube Playlist (Keynotes)](https://www.youtube.com/playlist?list=PLvvTyLx3m9oRW3K1OUka0LJWJEqR1tysD)

## Organizers

This Data Challenge was built in the open, with the goal of adding lasting artifacts to the community. _EvalRS_ was a collaboration between practitioners from industry and academia, who joined forces to make it happen:

* [Jacopo Tagliabue](https://www.linkedin.com/in/jacopotagliabue/), South Park Commons / NYU
* [Federico Bianchi](https://www.linkedin.com/in/federico-bianchi-3b7998121/), Stanford University
* [Tobias Schnabel](https://www.microsoft.com/en-us/research/people/toschnab/), Microsoft
* [Giuseppe Attanasio](https://www.linkedin.com/in/giuseppe-attanasio/), Bocconi University
* [Ciro Greco](https://www.linkedin.com/in/cirogreco/), South Park Commons
* [Gabriel de Souza P. Moreira](https://www.linkedin.com/in/gabrielspmoreira/), NVIDIA
* [Patrick John Chia](https://www.linkedin.com/in/patrick-john-chia/), Coveo

For inquiries, please reach out to the corresponding [author](https://www.linkedin.com/in/jacopotagliabue/).

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

_Challenge Review_

```
@misc{Tagliabue2023,
  doi = {10.1038/s42256-022-00606-0},
  url = {https://doi.org/10.1038/s42256-022-00606-0},
  author = {Tagliabue, Jacopo and Bianchi, Federico and Schnabel, Tobias and Attanasio, Giuseppe and Greco, Ciro and Moreira, Gabriel de Souza P. and Chia, Patrick John},
  title = {A challenge for rounded evaluation of recommender systems},
  publisher = {Nature Machine Intelligence},
  year = {2023}
}


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
1 | lyk | 1.70
2 | ML | 1.55
3 | fgiobergia | 1.33
4 | wwweiwei | 1.18
5 | Sunshine | 1.14

* First prize, _lyk team_ (3000 USD)
* Second prize, _ML team_ (1000 USD)

## Workshop Presentations

* [EvalRS Challenge Presentation](/workshop_slides/challenge_presentation_tagliabue.pdf) - [video](https://youtu.be/Jq2W454ypus)
* [Keynote by Prof. Jannach](/workshop_slides/keynote_jannach.pdf) - [video](https://youtu.be/VwFXQEv1kDM)
* [Keynote by Prof. Ekstrand](/workshop_slides/keynote_ekstrand.pdf) - [video](https://youtu.be/JT92-idNm0Q)
* [Best paper presentation, team ML](/workshop_slides/EvalRS_teamML.pdf)
* [Team wwweiwei](/workshop_slides/EvalRS_wwweiwei.pdf)
* [Team fgiobergia](/workshop_slides/EvalRS_fgiobergia.pdf)
* [Team Scrolls](/workshop_slides/EvalRS_Scrolls.pdf)

## Papers and Repositories

Team | Title | Paper | Repo |
--- | --- | --- | ---
wwweiwei | Track2Vec: Fairness Music Recommendation with a GPU-Free Customizable-Driven Framework | [paper](/final_papers/EvalRS2022_paper_582.pdf) [arxiv](https://arxiv.org/abs/2210.16590) | [code](https://github.com/wwweiwei/Track2Vec) 
fgiobergia | Triplet Losses-based Matrix Factorization for Robust Recommendations | [paper](/final_papers/EvalRS2022_paper_8348.pdf) [arxiv](https://arxiv.org/abs/2210.12098) | [code](https://github.com/fgiobergia/CIKM-evalRS-2022/) 
ML | Item-based Variational Auto-encoder for Fair Music Recommendation | [paper](/final_papers/EvalRS2022_paper_5248.pdf) [arxiv](https://arxiv.org/abs/2211.01333) | [code](https://github.com/ParkJinHyeock/evalRS-submission)
Scrolls | Bias Mitigation in Recommender Systems to Improve Diversity | [paper](/final_papers/EvalRS2022_paper_1487.pdf) | [code](https://github.com/fidelity/jurity/tree/evalrs/evalrs)
yao0510 | RecFormer: Personalized Temporal-Aware Transformer for Fair Music Recommendation | [paper](/final_papers/EvalRS2022_paper_4951.pdf) | [code](https://github.com/wywyWang/RecFormer) 
lyk | Diversity enhancement for Collaborative Filtering Recommendation | [paper](/final_papers/EvalRS2022_paper_5875.pdf) | [code](https://github.com/lazy2panda/evalrs2022_solution)

Selected papers also appear in the [Proceedings of the CIKM 2022 Workshops, co-located with 31st ACM International Conference on Information and Knowledge Management (CIKM 2022)](https://ceur-ws.org/Vol-3318/).
