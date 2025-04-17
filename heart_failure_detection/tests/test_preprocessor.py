"""
Tests for the data preprocessing module.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.preprocessor import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    """
    Test cases for the DataPreprocessor class.
    """
    
    def setUp(self):
        """
        Set up test data.
        """
        # Create a simple test dataframe
        self.test_data = pd.DataFrame({
            'Age': [45, 60, 55, 70, 50],
            'Sex': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'NYHA': ['I', 'II', 'III', 'IV', 'II'],
            'HTN': ['Yes', 'No', 'Yes', 'Yes', 'No'],
            'EF': [55, 40, 35, 30, 50],
            'HR': [80, 90, 100, 110, 85],
            'SBP': [120, 140, 160, 180, 130],
            'DBP': [80, 90, 100, 110, 85],
            'Creatinine': [0.8, 1.2, 1.5, 2.0, 1.0],
            'HF': [0, 1, 1, 1, 0]
        })
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor()
    
    def test_detect_outliers_iqr(self):
        """
        Test outlier detection using IQR method.
        """
        # Add an outlier
        test_data_with_outlier = self.test_data.copy()
        test_data_with_outlier.loc[5] = [45, 'Male', 'I', 'Yes', 55, 80, 300, 80, 0.8, 0]  # SBP outlier
        
        # Detect outliers
        outliers = self.preprocessor.detect_outliers_iqr(test_data_with_outlier)
        
        # Check if outlier is detected
        self.assertGreater(outliers['SBP'], 0)
    
    def test_cap_outliers(self):
        """
        Test outlier capping.
        """
        # Add an outlier
        test_data_with_outlier = self.test_data.copy()
        test_data_with_outlier.loc[5] = [45, 'Male', 'I', 'Yes', 55, 80, 300, 80, 0.8, 0]  # SBP outlier
        
        # Cap outliers
        capped_data = self.preprocessor.cap_outliers(test_data_with_outlier)
        
        # Check if outlier is capped
        self.assertLess(capped_data['SBP'].max(), 300)
    
    def test_encode_categorical_features(self):
        """
        Test categorical feature encoding.
        """
        # Encode categorical features
        encoded_data = self.preprocessor.encode_categorical_features(self.test_data)
        
        # Check if categorical columns are encoded
        self.assertNotIn('Sex', encoded_data.columns)
        self.assertNotIn('NYHA', encoded_data.columns)
        self.assertNotIn('HTN', encoded_data.columns)
        
        # Check if numerical columns are preserved
        self.assertIn('Age', encoded_data.columns)
        self.assertIn('EF', encoded_data.columns)
        self.assertIn('HR', encoded_data.columns)
        self.assertIn('SBP', encoded_data.columns)
        self.assertIn('DBP', encoded_data.columns)
        self.assertIn('Creatinine', encoded_data.columns)
        self.assertIn('HF', encoded_data.columns)
    
    def test_scale_features(self):
        """
        Test feature scaling.
        """
        # Encode categorical features first
        encoded_data = self.preprocessor.encode_categorical_features(self.test_data)
        
        # Scale features
        scaled_data = self.preprocessor.scale_features(encoded_data)
        
        # Check if numerical columns are scaled
        self.assertLessEqual(abs(scaled_data['Age'].mean()), 1.0)
        self.assertLessEqual(abs(scaled_data['EF'].mean()), 1.0)
        self.assertLessEqual(abs(scaled_data['HR'].mean()), 1.0)
        self.assertLessEqual(abs(scaled_data['SBP'].mean()), 1.0)
        self.assertLessEqual(abs(scaled_data['DBP'].mean()), 1.0)
        self.assertLessEqual(abs(scaled_data['Creatinine'].mean()), 1.0)
        
        # Check if target column is preserved
        self.assertEqual(scaled_data['HF'].tolist(), encoded_data['HF'].tolist())

if __name__ == '__main__':
    unittest.main()
