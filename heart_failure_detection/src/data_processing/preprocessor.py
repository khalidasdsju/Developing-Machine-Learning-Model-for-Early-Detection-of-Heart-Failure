"""
Data preprocessing module for the Heart Failure Detection project.
"""

import os
import numpy as np
import pandas as pd
import logging
import joblib
from scipy import stats
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split

from config.config import (
    PROCESSED_DATA_PATH, RANDOM_STATE, TEST_SIZE, TARGET_COLUMN,
    LABEL_ENCODING_COLUMNS, ONE_HOT_ENCODING_COLUMNS,
    SCALER_PATH, ENCODER_PATH
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Class for preprocessing the heart failure dataset.
    """
    
    def __init__(self):
        """Initialize the preprocessor with empty transformers."""
        self.label_encoder = LabelEncoder()
        self.scaler = RobustScaler()
        
    def detect_outliers_iqr(self, df):
        """
        Detect outliers using the IQR method.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            dict: Dictionary with column names as keys and number of outliers as values
        """
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_cols:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[column] = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            
        return outliers
    
    def transform_skewed_features(self, df):
        """
        Transform skewed features using Box-Cox transformation.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Transformed dataframe
        """
        df_transformed = df.copy()
        
        # Identify skewed features
        skewed_features = []
        for column in df.select_dtypes(include=[np.number]).columns:
            feature_skewness = stats.skew(df[column].dropna())
            _, p_value = stats.normaltest(df[column].dropna())
            
            if p_value < 0.05 or abs(feature_skewness) > 0.5:
                skewed_features.append(column)
        
        # Apply Box-Cox transformation to skewed features
        for column in skewed_features:
            if (df[column] > 0).all():  # Apply only if data is strictly positive
                df_transformed[column], _ = stats.boxcox(df[column] + 1)  # Add 1 to avoid log(0)
                
        logger.info(f"Transformed {len(skewed_features)} skewed features")
        return df_transformed
    
    def cap_outliers(self, df, lower_percentile=0.01, upper_percentile=0.99):
        """
        Cap outliers using winsorization.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            lower_percentile (float): Lower percentile for capping
            upper_percentile (float): Upper percentile for capping
            
        Returns:
            pandas.DataFrame: Dataframe with capped outliers
        """
        df_capped = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_cols:
            lower_limit = df[column].quantile(lower_percentile)
            upper_limit = df[column].quantile(upper_percentile)
            df_capped[column] = np.where(df[column] < lower_limit, lower_limit, df[column])
            df_capped[column] = np.where(df[column] > upper_limit, upper_limit, df[column])
            
        logger.info(f"Capped outliers in {len(numeric_cols)} numerical columns")
        return df_capped
    
    def encode_categorical_features(self, df, fit=True):
        """
        Encode categorical features using Label Encoding and One-Hot Encoding.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            fit (bool): Whether to fit the encoders or just transform
            
        Returns:
            pandas.DataFrame: Dataframe with encoded categorical features
        """
        df_encoded = df.copy()
        
        # Apply Label Encoding to binary categorical variables
        for col in LABEL_ENCODING_COLUMNS:
            if col in df.columns:
                if fit:
                    df_encoded[col] = self.label_encoder.fit_transform(df[col])
                else:
                    df_encoded[col] = self.label_encoder.transform(df[col])
        
        # Apply One-Hot Encoding to nominal categorical variables
        df_encoded = pd.get_dummies(df_encoded, columns=ONE_HOT_ENCODING_COLUMNS, drop_first=True)
        
        logger.info(f"Encoded {len(LABEL_ENCODING_COLUMNS)} columns with Label Encoding")
        logger.info(f"Encoded {len(ONE_HOT_ENCODING_COLUMNS)} columns with One-Hot Encoding")
        
        return df_encoded
    
    def scale_features(self, df, target_col=TARGET_COLUMN, fit=True):
        """
        Scale numerical features using RobustScaler.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            target_col (str): Name of the target column
            fit (bool): Whether to fit the scaler or just transform
            
        Returns:
            pandas.DataFrame: Dataframe with scaled features
        """
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Convert back to DataFrame
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Add target column back
        df_scaled = pd.concat([X_scaled_df, y], axis=1)
        
        logger.info(f"Scaled {X.shape[1]} features using RobustScaler")
        
        return df_scaled
    
    def preprocess_data(self, df, save_path=PROCESSED_DATA_PATH, save_transformers=True):
        """
        Preprocess the data by applying all preprocessing steps.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            save_path (str): Path to save the processed data
            save_transformers (bool): Whether to save the transformers
            
        Returns:
            tuple: Tuple containing (X_train, X_test, y_train, y_test)
        """
        try:
            logger.info("Starting data preprocessing")
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                logger.warning(f"Found {missing_values.sum()} missing values")
                # Handle missing values (simple imputation for now)
                df = df.fillna(df.mean())
                logger.info("Filled missing values with mean")
            
            # Transform skewed features
            df_transformed = self.transform_skewed_features(df)
            
            # Cap outliers
            df_capped = self.cap_outliers(df_transformed)
            
            # Encode categorical features
            df_encoded = self.encode_categorical_features(df_capped)
            
            # Scale features
            df_scaled = self.scale_features(df_encoded)
            
            # Split data into train and test sets
            X = df_scaled.drop(columns=[TARGET_COLUMN])
            y = df_scaled[TARGET_COLUMN]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
            )
            
            logger.info(f"Split data into train ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples) sets")
            
            # Save processed data
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                df_scaled.to_csv(save_path, index=False)
                logger.info(f"Saved processed data to {save_path}")
            
            # Save transformers
            if save_transformers:
                os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
                joblib.dump(self.scaler, SCALER_PATH)
                joblib.dump(self.label_encoder, ENCODER_PATH)
                logger.info(f"Saved transformers to {os.path.dirname(SCALER_PATH)}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def load_transformers(self):
        """
        Load the saved transformers.
        
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            self.scaler = joblib.load(SCALER_PATH)
            self.label_encoder = joblib.load(ENCODER_PATH)
            logger.info("Loaded transformers successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading transformers: {str(e)}")
            return False
    
    def preprocess_new_data(self, df):
        """
        Preprocess new data using the saved transformers.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Preprocessed dataframe
        """
        try:
            # Load transformers if not already loaded
            if not hasattr(self, 'scaler') or self.scaler is None:
                self.load_transformers()
            
            # Transform skewed features
            df_transformed = self.transform_skewed_features(df)
            
            # Cap outliers
            df_capped = self.cap_outliers(df_transformed)
            
            # Encode categorical features
            df_encoded = self.encode_categorical_features(df_capped, fit=False)
            
            # Scale features (without target column)
            if TARGET_COLUMN in df_encoded.columns:
                y = df_encoded[TARGET_COLUMN]
                X = df_encoded.drop(columns=[TARGET_COLUMN])
                X_scaled = self.scaler.transform(X)
                X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
                df_preprocessed = pd.concat([X_scaled_df, y], axis=1)
            else:
                X = df_encoded
                X_scaled = self.scaler.transform(X)
                df_preprocessed = pd.DataFrame(X_scaled, columns=X.columns)
            
            logger.info(f"Preprocessed new data with shape {df_preprocessed.shape}")
            return df_preprocessed
            
        except Exception as e:
            logger.error(f"Error preprocessing new data: {str(e)}")
            raise
