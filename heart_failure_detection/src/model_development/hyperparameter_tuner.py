"""
Hyperparameter tuning module for the Heart Failure Detection project.
"""

import logging
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import optuna
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from config.config import RANDOM_STATE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """
    Class for tuning hyperparameters of machine learning models.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            model_type (str): Type of model to tune ('random_forest', 'lightgbm', 'xgboost')
        """
        self.model_type = model_type
        
        # Define parameter grids for different models
        self.param_grids = {
            'random_forest': {
                'n_estimators': [100, 300, 500, 800, 1000, 1200],
                'max_depth': [5, 10, 15, 20, 25, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            },
            'lightgbm': {
                'boosting_type': ['gbdt', 'dart', 'goss'],
                'n_estimators': [100, 500, 1000, 1500, 2000],
                'learning_rate': [0.001, 0.01, 0.05, 0.1],
                'max_depth': [5, 10, 15, 20, 25, -1],
                'num_leaves': [31, 50, 100, 200],
                'min_child_samples': [5, 10, 20, 50]
            },
            'xgboost': {
                'n_estimators': [100, 500, 1000, 1500],
                'learning_rate': [0.001, 0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7, 9],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
        }
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=RANDOM_STATE)
        elif model_type == 'lightgbm':
            self.model = LGBMClassifier(random_state=RANDOM_STATE)
        elif model_type == 'xgboost':
            self.model = XGBClassifier(random_state=RANDOM_STATE)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def tune_with_randomized_search(self, X_train, y_train, param_grid=None, n_iter=100, cv=5, scoring='accuracy'):
        """
        Tune hyperparameters using RandomizedSearchCV.
        
        Args:
            X_train (pandas.DataFrame): Training features
            y_train (pandas.Series): Training target
            param_grid (dict): Parameter grid (if None, use default for model_type)
            n_iter (int): Number of parameter settings sampled
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            
        Returns:
            dict: Best parameters
        """
        try:
            # Use default param grid if not provided
            if param_grid is None:
                param_grid = self.param_grids[self.model_type]
            
            logger.info(f"Starting RandomizedSearchCV for {self.model_type} with {n_iter} iterations")
            
            # Initialize RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=1
            )
            
            # Fit RandomizedSearchCV
            random_search.fit(X_train, y_train)
            
            # Get best parameters and score
            best_params = random_search.best_params_
            best_score = random_search.best_score_
            
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best cross-validation score: {best_score:.4f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error in RandomizedSearchCV: {str(e)}")
            raise
    
    def tune_with_grid_search(self, X_train, y_train, param_grid=None, cv=5, scoring='accuracy'):
        """
        Tune hyperparameters using GridSearchCV.
        
        Args:
            X_train (pandas.DataFrame): Training features
            y_train (pandas.Series): Training target
            param_grid (dict): Parameter grid (if None, use default for model_type)
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            
        Returns:
            dict: Best parameters
        """
        try:
            # Use default param grid if not provided
            if param_grid is None:
                # For grid search, we need a smaller grid to avoid combinatorial explosion
                if self.model_type == 'random_forest':
                    param_grid = {
                        'n_estimators': [100, 500, 1000],
                        'max_depth': [10, 15, 20],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    }
                elif self.model_type == 'lightgbm':
                    param_grid = {
                        'n_estimators': [100, 500, 1000],
                        'learning_rate': [0.01, 0.1],
                        'max_depth': [10, 15, -1]
                    }
                elif self.model_type == 'xgboost':
                    param_grid = {
                        'n_estimators': [100, 500, 1000],
                        'learning_rate': [0.01, 0.1],
                        'max_depth': [3, 5, 7]
                    }
            
            logger.info(f"Starting GridSearchCV for {self.model_type}")
            
            # Initialize GridSearchCV
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
            
            # Fit GridSearchCV
            grid_search.fit(X_train, y_train)
            
            # Get best parameters and score
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best cross-validation score: {best_score:.4f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error in GridSearchCV: {str(e)}")
            raise
    
    def tune_with_optuna(self, X_train, y_train, cv=5, n_trials=100):
        """
        Tune hyperparameters using Optuna.
        
        Args:
            X_train (pandas.DataFrame): Training features
            y_train (pandas.Series): Training target
            cv (int): Number of cross-validation folds
            n_trials (int): Number of trials
            
        Returns:
            dict: Best parameters
        """
        try:
            from sklearn.model_selection import cross_val_score
            
            logger.info(f"Starting Optuna optimization for {self.model_type} with {n_trials} trials")
            
            # Define objective function for Optuna
            def objective(trial):
                # Define parameters based on model type
                if self.model_type == 'random_forest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
                        'max_depth': trial.suggest_int('max_depth', 5, 30),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                        'random_state': RANDOM_STATE
                    }
                    model = RandomForestClassifier(**params)
                
                elif self.model_type == 'lightgbm':
                    params = {
                        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
                        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1, log=True),
                        'max_bin': trial.suggest_int('max_bin', 200, 2000),
                        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                        'importance_type': trial.suggest_categorical('importance_type', ['split', 'gain']),
                        'random_state': RANDOM_STATE
                    }
                    model = LGBMClassifier(**params)
                
                elif self.model_type == 'xgboost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1, log=True),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'random_state': RANDOM_STATE
                    }
                    model = XGBClassifier(**params)
                
                # Perform cross-validation
                score = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
                
                return score
            
            # Create Optuna study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            # Get best parameters and score
            best_params = study.best_params
            best_score = study.best_value
            
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best cross-validation score: {best_score:.4f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error in Optuna optimization: {str(e)}")
            raise
