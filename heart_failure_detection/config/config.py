"""
Configuration settings for the Heart Failure Detection project.
"""

import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_data.sav")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.csv")

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")

# Data download settings
DATASET_URL = "https://drive.google.com/uc?id=1k401wxkiYxq5ngLkqSqRNZaWhDH25MEs"

# Model training settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "HF"
CATEGORICAL_COLUMNS = [
    'Sex', 'NYHA', 'HTN', 'DM', 'Smoker', 'DL', 'BA', 'CXR', 'RWMA', 'MI',
    'Chest_pain', 'ECG', 'ACS', 'Wall', 'MR', 'Thrombolysis'
]
LABEL_ENCODING_COLUMNS = [
    'Sex', 'NYHA', 'HTN', 'DM', 'Smoker', 'DL', 'BA', 'CXR', 'RWMA', 'MI',
    'Chest_pain'
]
ONE_HOT_ENCODING_COLUMNS = [
    'ECG', 'ACS', 'Wall', 'MR', 'Thrombolysis'
]
COLUMNS_TO_DROP = ['StudyID']

# Flask API settings
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = True
