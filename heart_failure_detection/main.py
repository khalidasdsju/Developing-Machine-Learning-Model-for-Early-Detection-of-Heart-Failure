"""
Main script for the Heart Failure Detection project.
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
from src.deployment.api import start_api
from config.config import PROCESSED_DATA_PATH, BEST_MODEL_PATH, TARGET_COLUMN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Heart Failure Detection Pipeline')
    
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'visualize', 'deploy'],
                        help='Mode to run the pipeline in')
    
    parser.add_argument('--tune', action='store_true',
                        help='Tune hyperparameters')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize data and model results')
    
    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['random_forest', 'lightgbm', 'xgboost'],
                        help='Type of model to train')
    
    return parser.parse_args()

def train_pipeline(args):
    """
    Run the training pipeline.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    try:
        logger.info("Starting training pipeline")
        
        # Load data
        data = load_dataset()
        if data is None:
            logger.error("Failed to load dataset. Exiting.")
            return
        
        # Create visualization directory
        os.makedirs('heart_failure_detection/visualizations', exist_ok=True)
        
        # Visualize data if requested
        if args.visualize:
            logger.info("Visualizing data")
            visualizer = DataVisualizer(save_dir='heart_failure_detection/visualizations')
            visualizer.plot_target_distribution(data, save=True)
            visualizer.plot_numerical_features(data, save=True)
            visualizer.plot_categorical_features(data, save=True)
            visualizer.plot_correlation_matrix(data, save=True)
            fig, skewed_features = visualizer.plot_skewed_features(data, save=True)
        
        # Preprocess data
        logger.info("Preprocessing data")
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(data)
        
        # Visualize transformed data if requested
        if args.visualize and 'skewed_features' in locals() and skewed_features:
            # Load processed data
            processed_data = pd.read_csv(PROCESSED_DATA_PATH)
            visualizer.plot_before_after_transformation(data, processed_data, skewed_features, save=True)
        
        # Tune hyperparameters if requested
        if args.tune:
            logger.info(f"Tuning hyperparameters for {args.model_type}")
            tuner = HyperparameterTuner(model_type=args.model_type)
            best_params = tuner.tune_with_optuna(X_train, y_train, n_trials=50)
        else:
            best_params = None
        
        # Train and evaluate models
        logger.info("Training and evaluating models")
        trainer = ModelTrainer()
        
        # Train all models and compare
        results_df = trainer.train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Train best model with tuned parameters
        if best_params:
            logger.info(f"Training best model ({args.model_type}) with tuned parameters")
            best_model = trainer.train_best_model(X_train, y_train, model_params=best_params)
        else:
            logger.info("Training best model with default parameters")
            best_model = trainer.train_best_model(X_train, y_train)
        
        # Evaluate best model
        evaluation = trainer.evaluate_model(best_model, X_test, y_test)
        
        # Save best model
        trainer.save_model(best_model)
        
        # Visualize model results if requested
        if args.visualize:
            logger.info("Visualizing model results")
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
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

def evaluate_pipeline():
    """
    Run the evaluation pipeline.
    """
    try:
        logger.info("Starting evaluation pipeline")
        
        # Load data
        data = load_dataset()
        if data is None:
            logger.error("Failed to load dataset. Exiting.")
            return
        
        # Preprocess data
        logger.info("Preprocessing data")
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(data, save_transformers=False)
        
        # Load model
        logger.info("Loading model")
        trainer = ModelTrainer()
        model = trainer.load_model()
        
        if model is None:
            logger.error("Failed to load model. Exiting.")
            return
        
        # Evaluate model
        logger.info("Evaluating model")
        evaluation = trainer.evaluate_model(model, X_test, y_test)
        
        # Print evaluation results
        logger.info(f"Accuracy: {evaluation['accuracy']:.4f}")
        logger.info(f"ROC AUC: {evaluation['roc_auc']:.4f}" if evaluation['roc_auc'] else "ROC AUC: Not available")
        logger.info(f"Classification Report:\n{pd.DataFrame(evaluation['classification_report']).T}")
        
        logger.info("Evaluation pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in evaluation pipeline: {str(e)}")
        raise

def visualize_pipeline():
    """
    Run the visualization pipeline.
    """
    try:
        logger.info("Starting visualization pipeline")
        
        # Load data
        data = load_dataset()
        if data is None:
            logger.error("Failed to load dataset. Exiting.")
            return
        
        # Create visualization directory
        os.makedirs('heart_failure_detection/visualizations', exist_ok=True)
        
        # Visualize data
        logger.info("Visualizing data")
        visualizer = DataVisualizer(save_dir='heart_failure_detection/visualizations')
        visualizer.plot_target_distribution(data, save=True)
        visualizer.plot_numerical_features(data, save=True)
        visualizer.plot_categorical_features(data, save=True)
        visualizer.plot_correlation_matrix(data, save=True)
        fig, skewed_features = visualizer.plot_skewed_features(data, save=True)
        
        # Preprocess data
        logger.info("Preprocessing data")
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(data, save_transformers=False)
        
        # Load model
        logger.info("Loading model")
        trainer = ModelTrainer()
        model = trainer.load_model()
        
        if model is None:
            logger.error("Failed to load model. Exiting.")
            return
        
        # Visualize model results
        logger.info("Visualizing model results")
        model_visualizer = ModelVisualizer(save_dir='heart_failure_detection/visualizations')
        
        # Plot confusion matrix
        y_pred = model.predict(X_test)
        model_visualizer.plot_confusion_matrix(y_test, y_pred, save=True)
        
        # Plot ROC curve and precision-recall curve if model supports predict_proba
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            model_visualizer.plot_roc_curve(y_test, y_pred_proba, save=True)
            model_visualizer.plot_precision_recall_curve(y_test, y_pred_proba, save=True)
        
        # Plot learning curve
        model_visualizer.plot_learning_curve(model, X_train, y_train, save=True)
        
        # Plot feature importance if model supports it
        if hasattr(model, 'feature_importances_'):
            model_visualizer.plot_feature_importance(model, X_train.columns, save=True)
        
        logger.info("Visualization pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in visualization pipeline: {str(e)}")
        raise

def deploy_pipeline():
    """
    Run the deployment pipeline.
    """
    try:
        logger.info("Starting deployment pipeline")
        
        # Check if model exists
        if not os.path.exists(BEST_MODEL_PATH):
            logger.error(f"Model not found at {BEST_MODEL_PATH}. Please train a model first.")
            return
        
        # Start API
        logger.info("Starting API")
        start_api()
        
    except Exception as e:
        logger.error(f"Error in deployment pipeline: {str(e)}")
        raise

def main():
    """
    Main function to run the pipeline.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Run pipeline based on mode
    if args.mode == 'train':
        train_pipeline(args)
    elif args.mode == 'evaluate':
        evaluate_pipeline()
    elif args.mode == 'visualize':
        visualize_pipeline()
    elif args.mode == 'deploy':
        deploy_pipeline()
    else:
        logger.error(f"Invalid mode: {args.mode}")

if __name__ == '__main__':
    main()
