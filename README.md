MetaWatch
==============================
[WIP] This project is currently under heavy prototyping stage and not that clean yet.

Meta analysis of meta compositions in Overwatch

Can currently analyze and extract live statistics from Overwatch replay videos such as:
- Health bars (Health, Armor, Shield, Death, DPS, HPS)
- Ultimate (Used, Ready)
- Player nicknames and heroes [WIP]
- Team nicknames
- Killfeed [WIP]
- Global action measure (Interest) according to the stats above
- Global action measure from the audio feed [EXPERIMENTAL AND WIP]

Usage
------------

1. Get a .mp4 replay video with the statistic banner on it.
2. Register the replay video using the register python script (src/data/raw...), it will move it into a file db in the data/raw folder as data/raw/<vid_UUID>/replay.mp4
3. Generate a set of matching masks using the PS layers (create a group, move the according slots onto the banner, export the layers using the export_layer .js script in the utils folder)
4. Paste the layers folders into the new <vid_UUID>/ folder
5. configure your mongoDB authentication in a local AUTH.cfg
6. Run the analysis using analyze... py file
7. Run the visualization Rmarkdown notebook to preprocess the raw data and visualize the results

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


###maitre mot de ce projet
https://xkcd.com/1319/
