# TWStock Price Predictor
A tool which predicts twstock price by using deep learning techniques.

## Folder structure
```
.
├── Makefile
├── Pipfile
├── Pipfile.lock
├── README.md
└── src
    ├── __init__.py
    ├── constants.py
    ├── data_fetcher.py
    ├── main.py
    ├── model.py
    ├── twstock
    │   ├── __init__.py
    │   ├── analytics.py
    │   ├── codes
    │   │   ├── __init__.py
    │   │   ├── codes.py
    │   │   ├── fetch.py
    │   │   ├── tpex_equities.csv
    │   │   └── twse_equities.csv
    │   ├── legacy.py
    │   ├── mock
    │   │   └── __init__.py
    │   ├── proxy.py
    │   ├── realtime.py
    │   └── stock.py
    ├── utils
    │   ├── __init__.py
    │   ├── data_utils.py
    │   ├── main_utils.py
    │   └── train_utils.py
    └── visualize.py
```

## How to run
1. Install `pipenv`: `$ pip install --user pipenv`
2. Install required package: `$ pipenv install`
3. Activate the python environment: `$ pipenv shell`
4. Fetch data (takes a long time): `$ make fetch-data`
5. Train model: `$ make train-model`

# Acknowledgement
Codes for fetching Taiwan stock prices (in `src/twstock/`) are from the following repository.
```
@misc{mlouielu,
  author = {Louie Lu},
  title = {twstock},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mlouielu/twstock}},
  commit = {bcd9b022aeb858549992b1251151c741e0c22d80}
}
```