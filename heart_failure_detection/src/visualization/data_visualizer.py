"""
Data visualization module for the Heart Failure Detection project.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, normaltest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataVisualizer:
    """
    Class for visualizing the heart failure dataset.
    """
    
    def __init__(self, save_dir=None):
        """
        Initialize the data visualizer.
        
        Args:
            save_dir (str): Directory to save visualizations
        """
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    
    def plot_target_distribution(self, df, target_col='HF', save=False):
        """
        Plot the distribution of the target variable.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            target_col (str): Name of the target column
            save (bool): Whether to save the plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        try:
            # Count the target values
            target_counts = df[target_col].value_counts()
            target_percentage = target_counts / target_counts.sum() * 100
            
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Pie chart
            colors = ['skyblue', 'lightcoral', 'red', 'olive'][:len(target_counts)]
            ax1.pie(
                target_counts,
                labels=target_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                explode=[0.1] * len(target_counts),
                shadow=True
            )
            ax1.set_title(f'{target_col} Status')
            ax1.axis('equal')
            
            # Bar chart
            bars = ax2.bar(
                target_counts.index,
                target_counts.values,
                color=colors,
                edgecolor='white',
                linewidth=1.2
            )
            
            # Annotate bars
            for bar, percentage in zip(bars, target_percentage):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f'{int(bar.get_height())} ({percentage:.1f}%)',
                    ha='center',
                    va='bottom',
                    fontsize=10
                )
            
            ax2.set_title(f'{target_col} Distribution', fontsize=14, weight='bold')
            ax2.set_ylabel('Count', fontsize=12)
            ax2.set_xlabel(target_col, fontsize=12)
            
            plt.tight_layout()
            
            # Save the plot if requested
            if save and self.save_dir:
                save_path = os.path.join(self.save_dir, f'{target_col}_distribution.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved target distribution plot to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting target distribution: {str(e)}")
            raise
    
    def plot_numerical_features(self, df, save=False):
        """
        Plot histograms and scatter plots for numerical features.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            save (bool): Whether to save the plots
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        try:
            # Select numerical columns
            num_cols = df.select_dtypes(include=[np.number]).columns
            
            # Create a figure with subplots
            fig, axes = plt.subplots(
                nrows=len(num_cols),
                ncols=2,
                figsize=(12, len(num_cols) * 4)
            )
            
            # Loop through numerical columns and plot
            for i, col in enumerate(num_cols):
                # Histogram with KDE
                sns.histplot(df[col], kde=True, ax=axes[i, 0])
                axes[i, 0].set_title(f'Histogram of {col}')
                
                # Detect outliers using IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Identify outliers
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                # Scatter plot with outliers highlighted
                sns.scatterplot(x=df.index, y=df[col], ax=axes[i, 1], color='skyblue')
                if not outliers.empty:
                    sns.scatterplot(x=outliers.index, y=outliers[col], ax=axes[i, 1], color='red')
                axes[i, 1].set_title(f'Scatter Plot of {col} with Outliers')
            
            plt.tight_layout()
            
            # Save the plot if requested
            if save and self.save_dir:
                save_path = os.path.join(self.save_dir, 'numerical_features.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved numerical features plot to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting numerical features: {str(e)}")
            raise
    
    def plot_categorical_features(self, df, save=False):
        """
        Plot bar charts for categorical features.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            save (bool): Whether to save the plots
            
        Returns:
            list: List of figure objects
        """
        try:
            # Select categorical columns
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            
            if len(cat_cols) == 0:
                logger.warning("No categorical columns found in the dataframe")
                return []
            
            figures = []
            
            # Loop through categorical columns and plot
            for col in cat_cols:
                # Create a new figure for each feature
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Count plot
                sns.countplot(x=df[col], palette=['skyblue', 'lightcoral', 'red', 'olive'], ax=ax)
                
                # Title
                ax.set_title(f'Bar Plot of {col}')
                
                # Calculate percentages
                total = len(df[col])
                for p in ax.patches:
                    height = p.get_height()
                    percentage = (height / total) * 100
                    ax.text(
                        p.get_x() + p.get_width() / 2,
                        height + 1,
                        f'{percentage:.2f}%',
                        ha='center',
                        va='bottom'
                    )
                
                # Rotate x-axis labels
                plt.xticks(rotation=90)
                
                plt.tight_layout()
                
                # Save the plot if requested
                if save and self.save_dir:
                    save_path = os.path.join(self.save_dir, f'{col}_distribution.png')
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved categorical feature plot to {save_path}")
                
                figures.append(fig)
            
            return figures
            
        except Exception as e:
            logger.error(f"Error plotting categorical features: {str(e)}")
            raise
    
    def plot_correlation_matrix(self, df, save=False):
        """
        Plot correlation matrix for numerical features.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            save (bool): Whether to save the plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        try:
            # Select numerical columns
            num_cols = df.select_dtypes(include=[np.number]).columns
            
            # Calculate correlation matrix
            corr_matrix = df[num_cols].corr()
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Plot heatmap
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                linewidths=0.5,
                vmin=-1,
                vmax=1
            )
            
            plt.title('Correlation Matrix', fontsize=16)
            plt.tight_layout()
            
            # Save the plot if requested
            if save and self.save_dir:
                save_path = os.path.join(self.save_dir, 'correlation_matrix.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved correlation matrix plot to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error plotting correlation matrix: {str(e)}")
            raise
    
    def plot_feature_distributions_by_target(self, df, target_col='HF', top_n=5, save=False):
        """
        Plot distributions of top features by target class.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            target_col (str): Name of the target column
            top_n (int): Number of top features to plot
            save (bool): Whether to save the plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        try:
            # Select numerical columns
            num_cols = df.select_dtypes(include=[np.number]).columns
            num_cols = [col for col in num_cols if col != target_col]
            
            # Calculate correlation with target
            correlations = []
            for col in num_cols:
                corr = df[col].corr(df[target_col])
                correlations.append((col, abs(corr)))
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: x[1], reverse=True)
            
            # Select top N features
            top_features = [x[0] for x in correlations[:top_n]]
            
            # Create figure
            fig, axes = plt.subplots(
                nrows=len(top_features),
                ncols=1,
                figsize=(10, len(top_features) * 4)
            )
            
            # Loop through top features and plot
            for i, feature in enumerate(top_features):
                # Plot distribution by target
                sns.boxplot(
                    x=target_col,
                    y=feature,
                    data=df,
                    palette='Set3',
                    ax=axes[i]
                )
                
                axes[i].set_title(f'Distribution of {feature} by {target_col}')
            
            plt.tight_layout()
            
            # Save the plot if requested
            if save and self.save_dir:
                save_path = os.path.join(self.save_dir, 'feature_distributions_by_target.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved feature distributions plot to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting feature distributions by target: {str(e)}")
            raise
    
    def plot_skewed_features(self, df, save=False):
        """
        Identify and plot skewed features.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            save (bool): Whether to save the plot
            
        Returns:
            tuple: Tuple containing (figure, skewed_features)
        """
        try:
            # Select numerical columns
            num_cols = df.select_dtypes(include=[np.number]).columns
            
            # Initialize lists for classification
            normal_dist = []
            right_skewed = []
            left_skewed = []
            non_normal_dist = []
            
            # Classify features based on skewness and normality
            for column in num_cols:
                # Calculate skewness
                feature_skewness = skew(df[column].dropna())
                
                # Check for normality
                _, p_value = normaltest(df[column].dropna())
                
                if abs(feature_skewness) < 0.5 and p_value > 0.05:
                    normal_dist.append(column)
                elif feature_skewness > 0.5:
                    right_skewed.append(column)
                elif feature_skewness < -0.5:
                    left_skewed.append(column)
                
                if p_value < 0.05:
                    non_normal_dist.append(column)
            
            # Combine skewed features
            skewed_features = right_skewed + left_skewed
            
            if not skewed_features:
                logger.info("No skewed features found")
                return None, []
            
            # Create figure for skewed features
            fig, axes = plt.subplots(
                nrows=len(skewed_features),
                ncols=1,
                figsize=(10, len(skewed_features) * 4)
            )
            
            # Handle case with only one skewed feature
            if len(skewed_features) == 1:
                axes = [axes]
            
            # Loop through skewed features and plot
            for i, feature in enumerate(skewed_features):
                # Plot histogram with KDE
                sns.histplot(df[feature], kde=True, ax=axes[i])
                
                # Add skewness value to title
                feature_skewness = skew(df[feature].dropna())
                axes[i].set_title(f'Distribution of {feature} (Skewness: {feature_skewness:.2f})')
            
            plt.tight_layout()
            
            # Save the plot if requested
            if save and self.save_dir:
                save_path = os.path.join(self.save_dir, 'skewed_features.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved skewed features plot to {save_path}")
            
            return fig, skewed_features
            
        except Exception as e:
            logger.error(f"Error plotting skewed features: {str(e)}")
            raise
    
    def plot_before_after_transformation(self, df_before, df_after, features, save=False):
        """
        Plot distributions before and after transformation.
        
        Args:
            df_before (pandas.DataFrame): Dataframe before transformation
            df_after (pandas.DataFrame): Dataframe after transformation
            features (list): List of features to plot
            save (bool): Whether to save the plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        try:
            # Create figure
            fig, axes = plt.subplots(
                nrows=len(features),
                ncols=2,
                figsize=(14, len(features) * 4)
            )
            
            # Loop through features and plot
            for i, feature in enumerate(features):
                # Plot before transformation
                sns.histplot(df_before[feature], kde=True, ax=axes[i, 0])
                axes[i, 0].set_title(f'Distribution of {feature} Before Transformation')
                
                # Plot after transformation
                sns.histplot(df_after[feature], kde=True, ax=axes[i, 1])
                axes[i, 1].set_title(f'Distribution of {feature} After Transformation')
            
            plt.tight_layout()
            
            # Save the plot if requested
            if save and self.save_dir:
                save_path = os.path.join(self.save_dir, 'before_after_transformation.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved before/after transformation plot to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting before/after transformation: {str(e)}")
            raise
