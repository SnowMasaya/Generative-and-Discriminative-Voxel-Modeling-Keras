# 3D classify model VoxceptionResNet and Search function develope by Keras
==============================

- 3d classify model
 - Voxel-Based Convnets for Classification. Model is the voxception-resnet as described in the [original paper](https://arxiv.org/pdf/1608.04236.pdf)
- 3d search function.
 - I use a [annoy](https://github.com/spotify/annoy) library

### Prerequisites

#### Library

linux

```
apt-get install pyenv
apt-get install virtualenv
```

Mac

```
brew install pyenv
brew install virtualenv
```

Make a virtual environment

```
pyenv install 3.6.0
pyenv rehash
pyenv local 3.6.0
virtualenv -p ~/.pyenv/versions/3.6.0/bin/python3.6 my_env
source my_env/bin/activate
```

Installing library

```
pip install -r requirements.txt
```

### Regist plotly

It used for 3D visualize

https://plot.ly/#/

#### Data


- Original data
  - [Model Net](http://modelnet.cs.princeton.edu/)


## Getting Started

- Preprocess data

```
python run_preprocess.py
```

- Train model

* You have to check data path

```
python run_trainer.py
```

- Regist search data

* You have to check model and data path

```
python run_search_regist.py
```

- Search Example

[Note book](https://github.com/SnowMasaya/Generative-and-Discriminative-Voxel-Modeling-Keras/blob/master/notebooks/3d_search_result.ipynb)

### PreTraining model

If you get the model, you use git lfs

Git LFS
- https://git-lfs.github.com/

```
git lfs pull
```

## Authors

[SnowMasaya](https://github.com/SnowMasaya)

## License

MIT

## References

- [Generative-and-Discriminative-Voxel-Modeling](https://github.com/varunkhare1234/Generative-and-Discriminative-Voxel-Modeling)

## Project Structure

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
