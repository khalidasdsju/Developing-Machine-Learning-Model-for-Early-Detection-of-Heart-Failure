"""
Model visualization module for the Heart Failure Detection project.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    auc, average_precision_score, roc_auc_score
)
from sklearn.model_selection import learning_curve

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelVisualizer:
    """
    Class for visualizing model performance.
    """
    
    def __init__(self, save_dir=None):
        """
        Initialize the model visualizer.
        
        Args:
            save_dir (str): Directory to save visualizations
        """
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, save=False, filename='confusion_matrix.png'):
        """
        Plot confusion matrix.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            labels (list): List of label names
            save (bool): Whether to save the plot
            filename (str): Filename for saved plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        try:
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create figure
            plt.figure(figsize=(8, 6))
            
            # Plot heatmap
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                cbar=False,
                xticklabels=labels,
                yticklabels=labels
            )
            
            plt.title('Confusion Matrix', fontsize=16)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            
            plt.tight_layout()
            
            # Save the plot if requested
            if save and self.save_dir:
                save_path = os.path.join(self.save_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved confusion matrix plot to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
            raise
    
    def plot_roc_curve(self, y_true, y_pred_proba, save=False, filename='roc_curve.png'):
        """
        Plot ROC curve.
        
        Args:
            y_true (array-like): True labels
            y_pred_proba (array-like): Predicted probabilities
            save (bool): Whether to save the plot
            filename (str): Filename for saved plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        try:
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Create figure
            plt.figure(figsize=(8, 6))
            
            # Plot ROC curve
            plt.plot(
                fpr, tpr,
                color='darkorange',
                lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})'
            )
            
            # Plot diagonal line
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
            plt.legend(loc='lower right')
            
            plt.tight_layout()
            
            # Save the plot if requested
            if save and self.save_dir:
                save_path = os.path.join(self.save_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved ROC curve plot to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error plotting ROC curve: {str(e)}")
            raise
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, save=False, filename='precision_recall_curve.png'):
        """
        Plot precision-recall curve.
        
        Args:
            y_true (array-like): True labels
            y_pred_proba (array-like): Predicted probabilities
            save (bool): Whether to save the plot
            filename (str): Filename for saved plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        try:
            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = average_precision_score(y_true, y_pred_proba)
            
            # Create figure
            plt.figure(figsize=(8, 6))
            
            # Plot precision-recall curve
            plt.plot(
                recall, precision,
                color='green',
                lw=2,
                label=f'Precision-Recall curve (AUC = {pr_auc:.4f})'
            )
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title('Precision-Recall Curve', fontsize=16)
            plt.legend(loc='lower left')
            
            plt.tight_layout()
            
            # Save the plot if requested
            if save and self.save_dir:
                save_path = os.path.join(self.save_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved precision-recall curve plot to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error plotting precision-recall curve: {str(e)}")
            raise
    
    def plot_learning_curve(self, estimator, X, y, cv=5, save=False, filename='learning_curve.png'):
        """
        Plot learning curve.
        
        Args:
            estimator (object): Trained model
            X (array-like): Features
            y (array-like): Target
            cv (int): Number of cross-validation folds
            save (bool): Whether to save the plot
            filename (str): Filename for saved plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        try:
            # Calculate learning curve
            train_sizes, train_scores, test_scores = learning_curve(
                estimator, X, y, cv=cv, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10)
            )
            
            # Calculate mean and standard deviation
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot learning curve
            plt.plot(
                train_sizes, train_mean,
                color='blue', marker='o',
                markersize=5, label='Training score'
            )
            plt.fill_between(
                train_sizes,
                train_mean + train_std,
                train_mean - train_std,
                alpha=0.15, color='blue'
            )
            
            plt.plot(
                train_sizes, test_mean,
                color='green', marker='s',
                markersize=5, label='Cross-validation score'
            )
            plt.fill_between(
                train_sizes,
                test_mean + test_std,
                test_mean - test_std,
                alpha=0.15, color='green'
            )
            
            plt.xlabel('Training Examples', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.title('Learning Curve', fontsize=16)
            plt.legend(loc='lower right')
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save the plot if requested
            if save and self.save_dir:
                save_path = os.path.join(self.save_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved learning curve plot to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error plotting learning curve: {str(e)}")
            raise
    
    def plot_feature_importance(self, model, feature_names, top_n=20, save=False, filename='feature_importance.png'):
        """
        Plot feature importance.
        
        Args:
            model (object): Trained model
            feature_names (list): List of feature names
            top_n (int): Number of top features to plot
            save (bool): Whether to save the plot
            filename (str): Filename for saved plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        try:
            # Check if model has feature_importances_ attribute
            if not hasattr(model, 'feature_importances_'):
                logger.warning("Model does not have feature_importances_ attribute")
                return None
            
            # Get feature importances
            importances = model.feature_importances_
            
            # Create DataFrame with feature names and importances
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            
            # Sort by importance
            feature_importance_df = feature_importance_df.sort_values(
                'Importance', ascending=False
            ).reset_index(drop=True)
            
            # Select top N features
            if len(feature_importance_df) > top_n:
                feature_importance_df = feature_importance_df.head(top_n)
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Plot feature importance
            sns.barplot(
                x='Importance',
                y='Feature',
                data=feature_importance_df,
                palette='viridis'
            )
            
            plt.title('Feature Importance', fontsize=16)
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            
            plt.tight_layout()
            
            # Save the plot if requested
            if save and self.save_dir:
                save_path = os.path.join(self.save_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved feature importance plot to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            raise
    
    def plot_model_comparison(self, results_df, metric='Accuracy', save=False, filename='model_comparison.png'):
        """
        Plot model comparison.
        
        Args:
            results_df (pandas.DataFrame): DataFrame with model results
            metric (str): Metric to compare
            save (bool): Whether to save the plot
            filename (str): Filename for saved plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        try:
            # Sort by metric
            sorted_df = results_df.sort_values(metric, ascending=False).reset_index(drop=True)
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot model comparison
            bars = sns.barplot(
                x='Model',
                y=metric,
                data=sorted_df,
                palette='viridis'
            )
            
            # Add value labels on top of bars
            for i, bar in enumerate(bars.patches):
                bars.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{bar.get_height():.4f}',
                    ha='center',
                    va='bottom',
                    fontsize=10
                )
            
            plt.title(f'Model Comparison by {metric}', fontsize=16)
            plt.xlabel('Model', fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save the plot if requested
            if save and self.save_dir:
                save_path = os.path.join(self.save_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved model comparison plot to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error plotting model comparison: {str(e)}")
            raise
