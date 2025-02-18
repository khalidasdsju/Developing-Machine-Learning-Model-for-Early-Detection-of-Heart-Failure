import os
from box.exceptions import BoxValueError
import yaml
from HFPredictions import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import mcnemar
import dvc.api




@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data
@ensure_annotations

def create_numpy_array(shape: tuple, value=0):
    """Create a numpy array of given shape filled with a specified value"""
    return np.full(shape, value)

def create_dataframe(data_dict: dict):
    """Create a pandas DataFrame from a dictionary"""
    return pd.DataFrame(data_dict)
def create_line_plot(x, y, title="Line Plot", xlabel="X-axis", ylabel="Y-axis"):
    """Create a simple line plot using matplotlib"""
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
def create_heatmap(df, title="Heatmap"):
    """Create a heatmap from a DataFrame"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.show()

def train_random_forest(X, y, test_size=0.2):
    """Train a Random Forest model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def apply_smote(X, y):
    """Apply SMOTE to balance the dataset"""
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train a LightGBM model"""
    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def train_tabnet(X_train, y_train, X_test, y_test):
    """Train a TabNet model"""
    model = TabNetClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def apply_fishers_exact_test(table):
    """Apply Fisher's Exact Test on a 2x2 contingency table"""
    _, p_value = fisher_exact(table)
    return p_value

def perform_mcnemar_test(table):
    """Perform McNemar's test"""
    result = mcnemar(table)
    return result.pvalue

def save_model_to_dvc(model, model_name):
    """Save model to DVC"""
    model_path = f"models/{model_name}.pkl"
    joblib.dump(model, model_path)
    dvc.api.add(model_path)
    return model_path

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())