import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import streamlit as st

class ModelEvaluator:
    """
    Comprehensive model evaluation utilities for PTSD prediction models.
    Provides clinical-relevant metrics and statistical analysis.
    """
    
    def __init__(self):
        self.evaluation_results = {}
        
    def comprehensive_evaluation(self, y_true, y_pred, y_pred_proba=None, model_name="Model"):
        """
        Perform comprehensive model evaluation with clinical metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            y_pred_proba: Predicted probabilities
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        
        results = {
            'model_name': model_name,
            'basic_metrics': self._calculate_basic_metrics(y_true, y_pred),
            'clinical_metrics': self._calculate_clinical_metrics(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        # Add probability-based metrics if available
        if y_pred_proba is not None:
            results['probability_metrics'] = self._calculate_probability_metrics(y_true, y_pred_proba)
            results['threshold_analysis'] = self._analyze_thresholds(y_true, y_pred_proba)
        
        self.evaluation_results[model_name] = results
        return results
    
    def _calculate_basic_metrics(self, y_true, y_pred):
        """Calculate basic classification metrics."""
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_score_macro': f1_score(y_true, y_pred, average='macro')
        }
    
    def _calculate_clinical_metrics(self, y_true, y_pred):
        """Calculate clinically relevant metrics for PTSD prediction."""
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Clinical metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # Additional clinical metrics
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Likelihood ratios
        lr_positive = sensitivity / false_positive_rate if false_positive_rate > 0 else float('inf')
        lr_negative = false_negative_rate / specificity if specificity > 0 else float('inf')
        
        # Diagnostic odds ratio
        diagnostic_or = (tp * tn) / (fp * fn) if (fp * fn) > 0 else float('inf')
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'positive_predictive_value': ppv,
            'negative_predictive_value': npv,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'likelihood_ratio_positive': lr_positive,
            'likelihood_ratio_negative': lr_negative,
            'diagnostic_odds_ratio': diagnostic_or,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    
    def _calculate_probability_metrics(self, y_true, y_pred_proba):
        """Calculate probability-based metrics."""
        
        try:
            auc_roc = roc_auc_score(y_true, y_pred_proba)
        except:
            auc_roc = None
            
        try:
            auc_pr = average_precision_score(y_true, y_pred_proba)
        except:
            auc_pr = None
        
        return {
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'brier_score': self._brier_score(y_true, y_pred_proba)
        }
    
    def _brier_score(self, y_true, y_pred_proba):
        """Calculate Brier score for probability calibration."""
        return np.mean((y_pred_proba - y_true) ** 2)
    
    def _analyze_thresholds(self, y_true, y_pred_proba, thresholds=None):
        """Analyze performance across different probability thresholds."""
        
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        threshold_results = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics for this threshold
            accuracy = accuracy_score(y_true, y_pred_thresh)
            precision = precision_score(y_true, y_pred_thresh, zero_division=0)
            recall = recall_score(y_true, y_pred_thresh, zero_division=0)
            f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
            
            threshold_results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        return pd.DataFrame(threshold_results)
    
    def cross_validation_evaluation(self, model, X, y, cv_folds=5, scoring_metrics=None):
        """
        Perform cross-validation evaluation with multiple metrics.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            cv_folds: Number of CV folds
            scoring_metrics: List of scoring metrics
            
        Returns:
            Dictionary with CV results
        """
        
        if scoring_metrics is None:
            scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for metric in scoring_metrics:
            try:
                scores = cross_val_score(model, X, y, cv=skf, scoring=metric)
                cv_results[metric] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores.tolist()
                }
            except Exception as e:
                st.warning(f"Could not calculate {metric}: {str(e)}")
                cv_results[metric] = {
                    'mean': None,
                    'std': None,
                    'scores': None
                }
        
        return cv_results
    
    def model_comparison(self, results_dict):
        """
        Compare multiple models and rank them.
        
        Args:
            results_dict: Dictionary with multiple model results
            
        Returns:
            DataFrame with model comparison
        """
        
        comparison_data = []
        
        for model_name, results in results_dict.items():
            basic_metrics = results.get('basic_metrics', {})
            clinical_metrics = results.get('clinical_metrics', {})
            prob_metrics = results.get('probability_metrics', {})
            
            row = {
                'Model': model_name,
                'Accuracy': basic_metrics.get('accuracy', 0),
                'Precision': basic_metrics.get('precision', 0),
                'Recall': basic_metrics.get('recall', 0),
                'F1-Score': basic_metrics.get('f1_score', 0),
                'Sensitivity': clinical_metrics.get('sensitivity', 0),
                'Specificity': clinical_metrics.get('specificity', 0),
                'PPV': clinical_metrics.get('positive_predictive_value', 0),
                'NPV': clinical_metrics.get('negative_predictive_value', 0),
                'AUC-ROC': prob_metrics.get('auc_roc', 0) if prob_metrics.get('auc_roc') else 0,
                'AUC-PR': prob_metrics.get('auc_pr', 0) if prob_metrics.get('auc_pr') else 0
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Add ranking
        comparison_df['Overall_Rank'] = comparison_df['F1-Score'].rank(ascending=False)
        comparison_df['Clinical_Rank'] = comparison_df['Sensitivity'].rank(ascending=False)
        
        return comparison_df.sort_values('Overall_Rank')
    
    def clinical_interpretation(self, results):
        """
        Provide clinical interpretation of model results.
        
        Args:
            results: Model evaluation results
            
        Returns:
            Dictionary with clinical interpretations
        """
        
        clinical_metrics = results.get('clinical_metrics', {})
        basic_metrics = results.get('basic_metrics', {})
        
        interpretation = {
            'overall_performance': self._interpret_overall_performance(basic_metrics),
            'clinical_utility': self._interpret_clinical_utility(clinical_metrics),
            'recommendations': self._generate_recommendations(results)
        }
        
        return interpretation
    
    def _interpret_overall_performance(self, basic_metrics):
        """Interpret overall model performance."""
        
        accuracy = basic_metrics.get('accuracy', 0)
        f1_score = basic_metrics.get('f1_score', 0)
        
        if accuracy >= 0.9 and f1_score >= 0.9:
            return "Excellent performance - suitable for clinical use"
        elif accuracy >= 0.8 and f1_score >= 0.8:
            return "Good performance - may be useful with careful validation"
        elif accuracy >= 0.7 and f1_score >= 0.7:
            return "Moderate performance - requires improvement before clinical use"
        else:
            return "Poor performance - not suitable for clinical use"
    
    def _interpret_clinical_utility(self, clinical_metrics):
        """Interpret clinical utility metrics."""
        
        sensitivity = clinical_metrics.get('sensitivity', 0)
        specificity = clinical_metrics.get('specificity', 0)
        ppv = clinical_metrics.get('positive_predictive_value', 0)
        npv = clinical_metrics.get('negative_predictive_value', 0)
        
        interpretations = []
        
        # Sensitivity interpretation
        if sensitivity >= 0.9:
            interpretations.append("High sensitivity - good at identifying PTSD cases")
        elif sensitivity >= 0.7:
            interpretations.append("Moderate sensitivity - may miss some PTSD cases")
        else:
            interpretations.append("Low sensitivity - likely to miss many PTSD cases")
        
        # Specificity interpretation
        if specificity >= 0.9:
            interpretations.append("High specificity - good at identifying non-PTSD cases")
        elif specificity >= 0.7:
            interpretations.append("Moderate specificity - some false positives expected")
        else:
            interpretations.append("Low specificity - many false positives expected")
        
        # PPV interpretation
        if ppv >= 0.8:
            interpretations.append("High positive predictive value - confident in positive predictions")
        elif ppv >= 0.6:
            interpretations.append("Moderate positive predictive value - some uncertainty in positive predictions")
        else:
            interpretations.append("Low positive predictive value - high uncertainty in positive predictions")
        
        return interpretations
    
    def _generate_recommendations(self, results):
        """Generate recommendations based on model performance."""
        
        recommendations = []
        clinical_metrics = results.get('clinical_metrics', {})
        basic_metrics = results.get('basic_metrics', {})
        
        sensitivity = clinical_metrics.get('sensitivity', 0)
        specificity = clinical_metrics.get('specificity', 0)
        accuracy = basic_metrics.get('accuracy', 0)
        
        # Performance-based recommendations
        if accuracy < 0.7:
            recommendations.append("Consider feature engineering or additional data collection")
            recommendations.append("Try ensemble methods or deep learning approaches")
        
        # Clinical balance recommendations
        if sensitivity < 0.7:
            recommendations.append("Adjust classification threshold to improve sensitivity")
            recommendations.append("Consider cost-sensitive learning to reduce false negatives")
        
        if specificity < 0.7:
            recommendations.append("Improve feature selection to reduce false positives")
            recommendations.append("Consider additional clinical validation")
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("Model performance is acceptable for clinical evaluation")
            recommendations.append("Consider prospective validation in clinical settings")
        
        return recommendations
    
    def export_evaluation_report(self, model_name):
        """
        Export comprehensive evaluation report.
        
        Args:
            model_name: Name of model to export
            
        Returns:
            Formatted evaluation report
        """
        
        if model_name not in self.evaluation_results:
            return None
        
        results = self.evaluation_results[model_name]
        
        report = f"""
# PTSD Prediction Model Evaluation Report
## Model: {model_name}

### Basic Performance Metrics
"""
        
        basic_metrics = results.get('basic_metrics', {})
        for metric, value in basic_metrics.items():
            report += f"- {metric.replace('_', ' ').title()}: {value:.4f}\n"
        
        report += "\n### Clinical Metrics\n"
        clinical_metrics = results.get('clinical_metrics', {})
        for metric, value in clinical_metrics.items():
            if isinstance(value, (int, float)):
                report += f"- {metric.replace('_', ' ').title()}: {value:.4f}\n"
        
        # Add clinical interpretation
        interpretation = self.clinical_interpretation(results)
        report += f"\n### Clinical Interpretation\n"
        report += f"**Overall Performance:** {interpretation['overall_performance']}\n\n"
        
        report += "**Clinical Utility:**\n"
        for util in interpretation['clinical_utility']:
            report += f"- {util}\n"
        
        report += "\n**Recommendations:**\n"
        for rec in interpretation['recommendations']:
            report += f"- {rec}\n"
        
        return report