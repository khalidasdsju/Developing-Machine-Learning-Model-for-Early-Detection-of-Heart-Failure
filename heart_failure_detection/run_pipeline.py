"""
Script to run the entire heart failure detection pipeline.
"""

import os
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from src.data_processing.data_loader import load_dataset
from src.data_processing.preprocessor import DataPreprocessor
from src.model_development.model_trainer import ModelTrainer
from src.model_development.hyperparameter_tuner import HyperparameterTuner
from src.visualization.data_visualizer import DataVisualizer
from src.visualization.model_visualizer import ModelVisualizer
from config.config import PROCESSED_DATA_PATH, BEST_MODEL_PATH, TARGET_COLUMN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline(tune=False, visualize=False):
    """
    Run the entire pipeline.
    
    Args:
        tune (bool): Whether to tune hyperparameters
        visualize (bool): Whether to visualize data and model results
    """
    try:
        logger.info("Starting heart failure detection pipeline")
        
        # Step 1: Load data
        logger.info("Step 1: Loading data")
        data = load_dataset()
        if data is None:
            logger.error("Failed to load dataset. Exiting.")
            return
        
        # Step 2: Visualize data (if requested)
        if visualize:
            logger.info("Step 2: Visualizing data")
            os.makedirs('heart_failure_detection/visualizations', exist_ok=True)
            visualizer = DataVisualizer(save_dir='heart_failure_detection/visualizations')
            visualizer.plot_target_distribution(data, save=True)
            visualizer.plot_numerical_features(data, save=True)
            visualizer.plot_categorical_features(data, save=True)
            visualizer.plot_correlation_matrix(data, save=True)
            fig, skewed_features = visualizer.plot_skewed_features(data, save=True)
        
        # Step 3: Preprocess data
        logger.info("Step 3: Preprocessing data")
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(data)
        
        # Step 4: Tune hyperparameters (if requested)
        if tune:
            logger.info("Step 4: Tuning hyperparameters")
            tuner = HyperparameterTuner(model_type='random_forest')
            best_params = tuner.tune_with_optuna(X_train, y_train, n_trials=50)
        else:
            best_params = None
        
        # Step 5: Train and evaluate models
        logger.info("Step 5: Training and evaluating models")
        trainer = ModelTrainer()
        results_df = trainer.train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Step 6: Train best model with tuned parameters
        logger.info("Step 6: Training best model")
        if best_params:
            best_model = trainer.train_best_model(X_train, y_train, model_params=best_params)
        else:
            best_model = trainer.train_best_model(X_train, y_train)
        
        # Step 7: Evaluate best model
        logger.info("Step 7: Evaluating best model")
        evaluation = trainer.evaluate_model(best_model, X_test, y_test)
        
        # Print evaluation results
        print("\nBest Model Evaluation Results:")
        print(f"Accuracy: {evaluation['accuracy']:.4f}")
        print(f"ROC AUC: {evaluation['roc_auc']:.4f}" if evaluation['roc_auc'] else "ROC AUC: Not available")
        print(f"\nClassification Report:\n{pd.DataFrame(evaluation['classification_report']).T}")
        
        # Step 8: Save best model
        logger.info("Step 8: Saving best model")
        trainer.save_model(best_model)
        
        # Step 9: Visualize model results (if requested)
        if visualize:
            logger.info("Step 9: Visualizing model results")
            model_visualizer = ModelVisualizer(save_dir='heart_failure_detection/visualizations')
            
            # Plot model comparison
            model_visualizer.plot_model_comparison(results_df, save=True)
            
            # Plot confusion matrix
            y_pred = best_model.predict(X_test)
            model_visualizer.plot_confusion_matrix(y_test, y_pred, save=True)
            
            # Plot ROC curve and precision-recall curve if model supports predict_proba
            if hasattr(best_model, 'predict_proba'):
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                model_visualizer.plot_roc_curve(y_test, y_pred_proba, save=True)
                model_visualizer.plot_precision_recall_curve(y_test, y_pred_proba, save=True)
            
            # Plot learning curve
            model_visualizer.plot_learning_curve(best_model, X_train, y_train, save=True)
            
            # Plot feature importance if model supports it
            if hasattr(best_model, 'feature_importances_'):
                model_visualizer.plot_feature_importance(best_model, X_train.columns, save=True)
        
        logger.info("Heart failure detection pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Heart Failure Detection Pipeline')
    parser.add_argument('--tune', action='store_true', help='Tune hyperparameters')
    parser.add_argument('--visualize', action='store_true', help='Visualize data and model results')
    args = parser.parse_args()
    
    # Run pipeline
    run_pipeline(tune=args.tune, visualize=args.visualize)
