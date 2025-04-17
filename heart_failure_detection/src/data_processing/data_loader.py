"""
Data loading module for the Heart Failure Detection project.
"""

import os
import gdown
import pandas as pd
import pyreadstat
import logging
from pathlib import Path

from config.config import RAW_DATA_PATH, DATASET_URL, COLUMNS_TO_DROP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_dataset(url=DATASET_URL, output_path=RAW_DATA_PATH):
    """
    Download the dataset from the provided URL.
    
    Args:
        url (str): URL to download the dataset from
        output_path (str): Path to save the downloaded dataset
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download the file
        logger.info(f"Downloading dataset from {url} to {output_path}")
        gdown.download(url, output_path, quiet=False)
        
        # Check if file exists
        if os.path.exists(output_path):
            logger.info(f"Dataset downloaded successfully to {output_path}")
            return True
        else:
            logger.error(f"Failed to download dataset to {output_path}")
            return False
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        return False

def load_dataset(file_path=RAW_DATA_PATH):
    """
    Load the dataset from the provided file path.
    
    Args:
        file_path (str): Path to the dataset file
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"Dataset file not found at {file_path}. Attempting to download...")
            download_success = download_dataset()
            if not download_success:
                logger.error("Failed to download dataset. Cannot proceed.")
                return None
        
        # Load the dataset
        logger.info(f"Loading dataset from {file_path}")
        data, meta = pyreadstat.read_sav(file_path)
        
        # Drop unnecessary columns
        if COLUMNS_TO_DROP:
            data = data.drop(columns=COLUMNS_TO_DROP)
            logger.info(f"Dropped columns: {COLUMNS_TO_DROP}")
        
        logger.info(f"Dataset loaded successfully with shape {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

if __name__ == "__main__":
    # Test the data loader
    data = load_dataset()
    if data is not None:
        print(f"Dataset shape: {data.shape}")
        print(f"Dataset columns: {data.columns.tolist()}")
        print(f"Dataset head:\n{data.head()}")
    else:
        print("Failed to load dataset.")
