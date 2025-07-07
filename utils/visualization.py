import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.decomposition import PCA
import plotly.figure_factory as ff

class Visualizer:
    """
    Visualization utilities for PTSD prediction analysis.
    Creates interactive plots for data exploration, model results, and feature analysis.
    """
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set1
        
    def plot_data_overview(self, df, target_column=None):
        """
        Create overview plots for dataset exploration.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            
        Returns:
            Dictionary of plotly figures
        """
        
        plots = {}
        
        # Data shape and types
        data_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Missing Values': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
        })
        
        # Missing values heatmap
        fig_missing = px.bar(
            data_info, 
            x='Column', 
            y='Missing %',
            title='Missing Values by Column',
            color='Missing %',
            color_continuous_scale='Reds'
        )
        fig_missing.update_layout(xaxis_tickangle=-45)
        plots['missing_values'] = fig_missing
        
        # Distribution plots for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Select top 6 numeric columns for visualization
            cols_to_plot = numeric_cols[:6]
            
            fig_dist = make_subplots(
                rows=2, cols=3,
                subplot_titles=cols_to_plot,
                specs=[[{"secondary_y": False}]*3]*2
            )
            
            for i, col in enumerate(cols_to_plot):
                row = i // 3 + 1
                col_idx = i % 3 + 1
                
                fig_dist.add_trace(
                    go.Histogram(x=df[col], name=col, showlegend=False),
                    row=row, col=col_idx
                )
            
            fig_dist.update_layout(
                title_text="Distribution of Numeric Features",
                height=600
            )
            plots['distributions'] = fig_dist
        
        # Target distribution if specified
        if target_column and target_column in df.columns:
            target_counts = df[target_column].value_counts()
            
            fig_target = px.pie(
                values=target_counts.values,
                names=target_counts.index,
                title=f'Distribution of {target_column}'
            )
            plots['target_distribution'] = fig_target
            
            # Feature correlation with target
            if len(numeric_cols) > 1:
                corr_with_target = df[numeric_cols].corrwith(df[target_column]).abs().sort_values(ascending=False)
                
                fig_corr = px.bar(
                    x=corr_with_target.values,
                    y=corr_with_target.index,
                    orientation='h',
                    title=f'Feature Correlation with {target_column}',
                    labels={'x': 'Absolute Correlation', 'y': 'Features'}
                )
                plots['target_correlation'] = fig_corr
        
        return plots
    
    def plot_pcl5_analysis(self, df):
        """
        Create PCL-5 specific analysis plots.
        
        Args:
            df: DataFrame with PCL-5 data
            
        Returns:
            Dictionary of plotly figures
        """
        
        plots = {}
        
        # Identify PCL-5 columns
        pcl5_cols = [col for col in df.columns if 'pcl5' in col.lower()]
        
        if len(pcl5_cols) == 0:
            return plots
        
        # PCL-5 subscale analysis
        subscales = {
            'Intrusive': [col for col in pcl5_cols if 'intrusive' in col.lower()],
            'Avoidance': [col for col in pcl5_cols if 'avoidance' in col.lower()],
            'Negative_Mood': [col for col in pcl5_cols if 'negative' in col.lower() or 'mood' in col.lower()],
            'Hyperarousal': [col for col in pcl5_cols if 'hyper' in col.lower() or 'arousal' in col.lower()]
        }
        
        # If subscale columns exist, create radar chart
        subscale_data = []
        for subscale, cols in subscales.items():
            if cols:
                mean_score = df[cols].mean().mean()
                subscale_data.append({'Subscale': subscale, 'Mean_Score': mean_score})
        
        if subscale_data:
            subscale_df = pd.DataFrame(subscale_data)
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=subscale_df['Mean_Score'],
                theta=subscale_df['Subscale'],
                fill='toself',
                name='PCL-5 Subscales'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5]
                    )),
                title="PCL-5 Subscale Profile"
            )
            plots['pcl5_radar'] = fig_radar
        
        # PCL-5 total score distribution
        total_score_cols = [col for col in pcl5_cols if 'total' in col.lower()]
        if total_score_cols:
            fig_total = px.histogram(
                df, 
                x=total_score_cols[0],
                title='PCL-5 Total Score Distribution',
                nbins=20
            )
            plots['pcl5_total_dist'] = fig_total
        
        return plots
    
    def plot_model_results(self, results_dict):
        """
        Create comprehensive model performance visualizations.
        
        Args:
            results_dict: Dictionary with model results
            
        Returns:
            Dictionary of plotly figures
        """
        
        plots = {}
        
        # Model comparison metrics
        metrics_data = []
        for model_name, results in results_dict.items():
            metrics_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'AUC': results.get('auc', 0)
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Metrics comparison bar chart
        fig_metrics = px.bar(
            metrics_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
            x='Model',
            y='Score',
            color='Metric',
            barmode='group',
            title='Model Performance Comparison',
            range_y=[0, 1]
        )
        fig_metrics.update_layout(xaxis_tickangle=-45)
        plots['metrics_comparison'] = fig_metrics
        
        # Cross-validation scores
        cv_data = []
        for model_name, results in results_dict.items():
            cv_data.append({
                'Model': model_name,
                'CV_Mean': results['cv_mean'],
                'CV_Std': results['cv_std']
            })
        
        cv_df = pd.DataFrame(cv_data)
        
        fig_cv = px.bar(
            cv_df,
            x='Model',
            y='CV_Mean',
            error_y='CV_Std',
            title='Cross-Validation Scores',
            range_y=[0, 1]
        )
        plots['cv_scores'] = fig_cv
        
        # ROC curves
        fig_roc = go.Figure()
        
        for model_name, results in results_dict.items():
            if results.get('probabilities') is not None:
                fpr, tpr, _ = roc_curve(results['actual'], results['probabilities'])
                auc_score = results.get('auc', 0)
                
                fig_roc.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'{model_name} (AUC = {auc_score:.3f})'
                ))
        
        # Add diagonal line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random Classifier',
            showlegend=False
        ))
        
        fig_roc.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=500
        )
        plots['roc_curves'] = fig_roc
        
        # Confusion matrices
        for model_name, results in results_dict.items():
            cm = results['confusion_matrix']
            
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title=f'Confusion Matrix - {model_name}',
                labels=dict(x="Predicted", y="Actual"),
                x=['No PTSD', 'PTSD'],
                y=['No PTSD', 'PTSD']
            )
            plots[f'confusion_matrix_{model_name}'] = fig_cm
        
        return plots
    
    def plot_feature_importance(self, feature_importance, feature_names, model_name):
        """
        Create feature importance visualization.
        
        Args:
            feature_importance: Array of feature importance scores
            feature_names: List of feature names
            model_name: Name of the model
            
        Returns:
            Plotly figure
        """
        
        if feature_importance is None:
            return None
        
        # Create DataFrame and sort by importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)
        
        # Take top 20 features for visualization
        top_features = importance_df.tail(20)
        
        fig = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Feature Importance - {model_name}',
            labels={'Importance': 'Feature Importance Score'}
        )
        
        fig.update_layout(height=600)
        
        return fig
    
    def plot_prediction_analysis(self, y_true, y_pred, y_pred_proba=None):
        """
        Create prediction analysis plots.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of plotly figures
        """
        
        plots = {}
        
        # Prediction distribution
        pred_df = pd.DataFrame({
            'True_Label': y_true,
            'Predicted_Label': y_pred,
            'Correct': y_true == y_pred
        })
        
        fig_pred_dist = px.histogram(
            pred_df,
            x='Predicted_Label',
            color='Correct',
            barmode='group',
            title='Prediction Distribution'
        )
        plots['prediction_distribution'] = fig_pred_dist
        
        # Probability distribution if available
        if y_pred_proba is not None:
            prob_df = pd.DataFrame({
                'Probability': y_pred_proba,
                'True_Label': y_true
            })
            
            fig_prob = px.histogram(
                prob_df,
                x='Probability',
                color='True_Label',
                barmode='overlay',
                title='Prediction Probability Distribution',
                opacity=0.7
            )
            plots['probability_distribution'] = fig_prob
            
            # Calibration plot
            from sklearn.calibration import calibration_curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=10
            )
            
            fig_calib = go.Figure()
            fig_calib.add_trace(go.Scatter(
                x=mean_predicted_value,
                y=fraction_of_positives,
                mode='markers+lines',
                name='Model'
            ))
            fig_calib.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Perfect Calibration'
            ))
            
            fig_calib.update_layout(
                title='Calibration Plot',
                xaxis_title='Mean Predicted Probability',
                yaxis_title='Fraction of Positives'
            )
            plots['calibration'] = fig_calib
        
        return plots
    
    def plot_data_pca(self, X, y, feature_names=None):
        """
        Create PCA visualization of the data.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            
        Returns:
            Plotly figure
        """
        
        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create DataFrame for plotting
        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Target': y
        })
        
        fig = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='Target',
            title=f'PCA Visualization (Explained Variance: {pca.explained_variance_ratio_.sum():.3f})',
            labels={
                'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.3f})',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.3f})'
            }
        )
        
        return fig
    
    def create_model_summary_table(self, results_dict):
        """
        Create a summary table of model performance.
        
        Args:
            results_dict: Dictionary with model results
            
        Returns:
            DataFrame with model summary
        """
        
        summary_data = []
        for model_name, results in results_dict.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{results['accuracy']:.3f}",
                'Precision': f"{results['precision']:.3f}",
                'Recall': f"{results['recall']:.3f}",
                'F1-Score': f"{results['f1_score']:.3f}",
                'AUC': f"{results.get('auc', 0):.3f}" if results.get('auc') else 'N/A',
                'CV Mean': f"{results['cv_mean']:.3f}",
                'CV Std': f"{results['cv_std']:.3f}"
            })
        
        return pd.DataFrame(summary_data)