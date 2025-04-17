"""
Model training module for the Heart Failure Detection project.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from config.config import BEST_MODEL_PATH, RANDOM_STATE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Class for training and evaluating machine learning models for heart failure detection.
    """
    
    def __init__(self):
        """Initialize the model trainer with a set of models."""
        self.models = {
            'Logistic Regression': LogisticRegression(
                C=0.3, solver='liblinear', penalty='l1', class_weight='balanced', random_state=RANDOM_STATE
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5, weights='distance', algorithm='ball_tree', leaf_size=20, p=2
            ),
            'Naive Bayes': GaussianNB(var_smoothing=1e-8),
            'Support Vector Machine': SVC(
                C=5.0, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=RANDOM_STATE
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_split=3, min_samples_leaf=1,
                class_weight='balanced', random_state=RANDOM_STATE
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=300, max_depth=15, min_samples_split=3, min_samples_leaf=1,
                class_weight='balanced', random_state=RANDOM_STATE
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=300, learning_rate=0.1, max_depth=15, random_state=RANDOM_STATE
            ),
            'XGBoost': XGBClassifier(
                n_estimators=300, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE
            ),
            'CatBoost': CatBoostClassifier(
                iterations=300, learning_rate=0.1, depth=5, random_state=RANDOM_STATE, verbose=0
            )
        }
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0.0
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test, cv=5):
        """
        Train and evaluate multiple models.
        
        Args:
            X_train (pandas.DataFrame): Training features
            X_test (pandas.DataFrame): Testing features
            y_train (pandas.Series): Training target
            y_test (pandas.Series): Testing target
            cv (int): Number of cross-validation folds
            
        Returns:
            pandas.DataFrame: DataFrame with model evaluation results
        """
        results = []
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name}...")
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Calculate cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Calculate ROC AUC if the model supports predict_proba
                if hasattr(model, 'predict_proba'):
                    try:
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        roc_auc = roc_auc_score(y_test, y_pred_proba)
                    except:
                        roc_auc = None
                else:
                    roc_auc = None
                
                # Store results
                results.append({
                    'Model': model_name,
                    'Accuracy': accuracy,
                    'CV Mean': cv_mean,
                    'CV Std': cv_std,
                    'ROC AUC': roc_auc
                })
                
                # Update best model
                if cv_mean > self.best_score:
                    self.best_score = cv_mean
                    self.best_model = model
                    self.best_model_name = model_name
                
                logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, CV Mean: {cv_mean:.4f}, CV Std: {cv_std:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        logger.info(f"Best model: {self.best_model_name} with CV Mean: {self.best_score:.4f}")
        
        return results_df
    
    def train_best_model(self, X_train, y_train, model_params=None):
        """
        Train the best model with optional hyperparameters.
        
        Args:
            X_train (pandas.DataFrame): Training features
            y_train (pandas.Series): Training target
            model_params (dict): Optional hyperparameters for the model
            
        Returns:
            object: Trained model
        """
        try:
            # If best model is not set, use Random Forest as default
            if self.best_model is None:
                self.best_model_name = 'Random Forest'
                self.best_model = RandomForestClassifier(
                    n_estimators=1000, max_depth=15, min_samples_split=3, min_samples_leaf=1,
                    class_weight='balanced', random_state=RANDOM_STATE
                )
            
            # Update model parameters if provided
            if model_params is not None:
                self.best_model.set_params(**model_params)
            
            logger.info(f"Training best model ({self.best_model_name}) with parameters: {self.best_model.get_params()}")
            
            # Train the model
            self.best_model.fit(X_train, y_train)
            
            return self.best_model
            
        except Exception as e:
            logger.error(f"Error training best model: {str(e)}")
            raise
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate a model on the test set.
        
        Args:
            model (object): Trained model
            X_test (pandas.DataFrame): Testing features
            y_test (pandas.Series): Testing target
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Calculate ROC AUC if the model supports predict_proba
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    roc_auc = None
            else:
                roc_auc = None
            
            # Store evaluation results
            evaluation = {
                'accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'roc_auc': roc_auc
            }
            
            logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f if roc_auc else None}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, model=None, model_path=BEST_MODEL_PATH):
        """
        Save the model to disk.
        
        Args:
            model (object): Model to save (if None, save best_model)
            model_path (str): Path to save the model
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        try:
            # Use best_model if model is not provided
            if model is None:
                model = self.best_model
            
            # Check if model exists
            if model is None:
                logger.error("No model to save")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save the model
            joblib.dump(model, model_path)
            
            logger.info(f"Model saved to {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path=BEST_MODEL_PATH):
        """
        Load a model from disk.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            object: Loaded model
        """
        try:
            # Check if file exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at {model_path}")
                return None
            
            # Load the model
            model = joblib.load(model_path)
            
            logger.info(f"Model loaded from {model_path}")
            
            # Update best_model
            self.best_model = model
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
