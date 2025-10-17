import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_iris_data():
    iris = fetch_ucirepo(id=53)
    X = pd.DataFrame(iris.data.features)
    y = pd.DataFrame(iris.data.targets)
    y.columns = ['species']
    y['species'] = y['species'].str.replace('Iris-', '', regex=False)
    return X.to_numpy(), y.to_numpy().flatten()

def load_wine_data():
    wine = fetch_ucirepo(id=109)
    X = pd.DataFrame(wine.data.features)
    y = pd.DataFrame(wine.data.targets)
    y.columns = ['class']
    return X.to_numpy(), y.to_numpy().flatten()
