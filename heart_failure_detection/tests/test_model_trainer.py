"""
Tests for the model training module.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_development.model_trainer import ModelTrainer

class TestModelTrainer(unittest.TestCase):
    """
    Test cases for the ModelTrainer class.
    """
    
    def setUp(self):
        """
        Set up test data.
        """
        # Create a simple synthetic dataset
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        
        # Convert to DataFrame
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        self.y = pd.Series(y, name='target')
        
        # Split into train and test sets
        self.X_train = self.X[:80]
        self.X_test = self.X[80:]
        self.y_train = self.y[:80]
        self.y_test = self.y[80:]
        
        # Initialize model trainer
        self.trainer = ModelTrainer()
    
    def test_train_and_evaluate_models(self):
        """
        Test training and evaluating multiple models.
        """
        # Train and evaluate models
        results_df = self.trainer.train_and_evaluate_models(
            self.X_train, self.X_test, self.y_train, self.y_test, cv=2
        )
        
        # Check if results are returned
        self.assertIsNotNone(results_df)
        self.assertGreater(len(results_df), 0)
        
        # Check if required columns are present
        self.assertIn('Model', results_df.columns)
        self.assertIn('Accuracy', results_df.columns)
        self.assertIn('CV Mean', results_df.columns)
        self.assertIn('CV Std', results_df.columns)
        
        # Check if best model is set
        self.assertIsNotNone(self.trainer.best_model)
        self.assertIsNotNone(self.trainer.best_model_name)
        self.assertGreater(self.trainer.best_score, 0.0)
    
    def test_train_best_model(self):
        """
        Test training the best model.
        """
        # Train best model
        model = self.trainer.train_best_model(self.X_train, self.y_train)
        
        # Check if model is trained
        self.assertIsNotNone(model)
        
        # Check if model can make predictions
        y_pred = model.predict(self.X_test)
        self.assertEqual(len(y_pred), len(self.y_test))
    
    def test_evaluate_model(self):
        """
        Test evaluating a model.
        """
        # Train a model
        model = self.trainer.train_best_model(self.X_train, self.y_train)
        
        # Evaluate model
        evaluation = self.trainer.evaluate_model(model, self.X_test, self.y_test)
        
        # Check if evaluation results are returned
        self.assertIsNotNone(evaluation)
        self.assertIn('accuracy', evaluation)
        self.assertIn('classification_report', evaluation)
        self.assertIn('confusion_matrix', evaluation)
    
    def test_save_and_load_model(self):
        """
        Test saving and loading a model.
        """
        # Create a temporary file path
        temp_model_path = 'temp_model.pkl'
        
        # Train a model
        model = self.trainer.train_best_model(self.X_train, self.y_train)
        
        # Save model
        save_success = self.trainer.save_model(model, model_path=temp_model_path)
        self.assertTrue(save_success)
        self.assertTrue(os.path.exists(temp_model_path))
        
        # Load model
        loaded_model = self.trainer.load_model(model_path=temp_model_path)
        self.assertIsNotNone(loaded_model)
        
        # Check if loaded model can make predictions
        y_pred = loaded_model.predict(self.X_test)
        self.assertEqual(len(y_pred), len(self.y_test))
        
        # Clean up
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

if __name__ == '__main__':
    unittest.main()
