import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from models.ml_models import PTSDMLModels
from utils.evaluation import ModelEvaluator
from utils.visualization import Visualizer
from database.database_manager import DatabaseManager
import joblib
import os

# Configure page
st.set_page_config(
    page_title="Model Training - PTSD ML Platform",
    page_icon="🤖",
    layout="wide"
)

def main():
    st.title("🤖 Model Training & Evaluation")
    st.markdown("Train machine learning models for PTSD prediction and evaluate their performance.")
    
    # Initialize components
    if 'ml_models' not in st.session_state:
        st.session_state.ml_models = PTSDMLModels()
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = ModelEvaluator()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer()
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    # Check if data is available
    if 'features' not in st.session_state or 'target' not in st.session_state:
        st.warning("⚠️ No processed data available. Please upload and process data first in the Data Upload page.")
        return
    
    X = st.session_state.features
    y = st.session_state.target
    
    st.success(f"✅ Data ready for training: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Training configuration
    st.header("⚙️ Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider(
            "Test Set Size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Proportion of data to use for testing"
        )
    
    with col2:
        cv_folds = st.slider(
            "Cross-Validation Folds",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of folds for cross-validation"
        )
    
    with col3:
        random_state = st.number_input(
            "Random State",
            min_value=0,
            max_value=1000,
            value=42,
            help="Seed for reproducible results"
        )
    
    # Model selection
    st.header("🔧 Model Selection")
    
    available_models = ['SVM', 'ANN', 'Decision_Tree', 'Gaussian_NB', 'Random_Forest', 'Gradient_Boosting']
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_models = st.multiselect(
            "Select models to train:",
            available_models,
            default=available_models,
            help="Choose which models to train and compare"
        )
    
    with col2:
        quick_train = st.checkbox(
            "Quick Training",
            value=False,
            help="Use faster settings for testing (may reduce accuracy)"
        )
    
    # Training button
    if st.button("🚀 Start Training", type="primary"):
        if not selected_models:
            st.warning("Please select at least one model to train.")
            return
        
        # Training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            with st.spinner("Training models..."):
                # Update ML models to only include selected models
                original_models = st.session_state.ml_models.models.copy()
                st.session_state.ml_models.models = {
                    name: model for name, model in original_models.items() 
                    if name in selected_models
                }
                
                # Train models
                status_text.text("Training machine learning models...")
                results = st.session_state.ml_models.train_models(
                    X, y, 
                    test_size=test_size, 
                    cv_folds=cv_folds
                )
                
                progress_bar.progress(0.7)
                
                # Comprehensive evaluation
                status_text.text("Evaluating model performance...")
                evaluation_results = {}
                
                for model_name, result in results.items():
                    evaluation = st.session_state.evaluator.comprehensive_evaluation(
                        result['actual'],
                        result['predictions'],
                        result.get('probabilities'),
                        model_name
                    )
                    evaluation_results[model_name] = evaluation
                
                progress_bar.progress(0.9)
                
                # Save results to database
                status_text.text("Saving results to database...")
                for model_name, result in results.items():
                    model_data = {
                        'accuracy': result['accuracy'],
                        'precision': result['precision'],
                        'recall': result['recall'],
                        'f1_score': result['f1_score'],
                        'auc': result.get('auc'),
                        'cv_mean': result['cv_mean'],
                        'cv_std': result['cv_std'],
                        'confusion_matrix': result['confusion_matrix'],
                        'training_data_size': X.shape[0],
                        'feature_count': X.shape[1]
                    }
                    st.session_state.db_manager.save_model_results(model_name, model_data)
                
                progress_bar.progress(1.0)
                status_text.text("Training completed!")
                
                # Store results in session state
                st.session_state.model_results = results
                st.session_state.evaluation_results = evaluation_results
                st.session_state.models_trained = True
                
                st.success("✅ Model training completed successfully!")
                
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            return
    
    # Display results if available
    if 'model_results' in st.session_state and st.session_state.model_results:
        st.header("📊 Training Results")
        
        results = st.session_state.model_results
        
        # Performance summary table
        st.subheader("Performance Summary")
        summary_df = st.session_state.visualizer.create_model_summary_table(results)
        st.dataframe(summary_df, use_container_width=True)
        
        # Best model highlight
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_accuracy = results[best_model]['accuracy']
        
        st.info(f"🏆 **Best Model:** {best_model} with {best_accuracy:.3f} accuracy")
        
        # Visualizations
        st.subheader("Performance Visualizations")
        
        # Generate plots
        plots = st.session_state.visualizer.plot_model_results(results)
        
        # Display plots in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Metrics Comparison", "🎯 ROC Curves", "📊 Cross-Validation", "🔍 Confusion Matrices"])
        
        with tab1:
            if 'metrics_comparison' in plots:
                st.plotly_chart(plots['metrics_comparison'], use_container_width=True)
        
        with tab2:
            if 'roc_curves' in plots:
                st.plotly_chart(plots['roc_curves'], use_container_width=True)
        
        with tab3:
            if 'cv_scores' in plots:
                st.plotly_chart(plots['cv_scores'], use_container_width=True)
        
        with tab4:
            # Display confusion matrices
            cm_cols = st.columns(len(results))
            for i, (model_name, result) in enumerate(results.items()):
                with cm_cols[i]:
                    if f'confusion_matrix_{model_name}' in plots:
                        st.plotly_chart(plots[f'confusion_matrix_{model_name}'], use_container_width=True)
        
        # Detailed evaluation
        st.header("🔬 Detailed Evaluation")
        
        selected_model = st.selectbox(
            "Select model for detailed analysis:",
            list(results.keys()),
            index=list(results.keys()).index(best_model) if best_model in results else 0
        )
        
        if selected_model and 'evaluation_results' in st.session_state:
            evaluation = st.session_state.evaluation_results.get(selected_model, {})
            
            # Clinical metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Clinical Metrics")
                clinical_metrics = evaluation.get('clinical_metrics', {})
                
                metrics_df = pd.DataFrame([
                    {'Metric': 'Sensitivity (Recall)', 'Value': f"{clinical_metrics.get('sensitivity', 0):.3f}"},
                    {'Metric': 'Specificity', 'Value': f"{clinical_metrics.get('specificity', 0):.3f}"},
                    {'Metric': 'Positive Predictive Value', 'Value': f"{clinical_metrics.get('positive_predictive_value', 0):.3f}"},
                    {'Metric': 'Negative Predictive Value', 'Value': f"{clinical_metrics.get('negative_predictive_value', 0):.3f}"},
                    {'Metric': 'False Positive Rate', 'Value': f"{clinical_metrics.get('false_positive_rate', 0):.3f}"},
                    {'Metric': 'False Negative Rate', 'Value': f"{clinical_metrics.get('false_negative_rate', 0):.3f}"}
                ])
                
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("Confusion Matrix Details")
                
                cm_data = clinical_metrics
                cm_df = pd.DataFrame([
                    {'Category': 'True Positives', 'Count': cm_data.get('true_positives', 0)},
                    {'Category': 'True Negatives', 'Count': cm_data.get('true_negatives', 0)},
                    {'Category': 'False Positives', 'Count': cm_data.get('false_positives', 0)},
                    {'Category': 'False Negatives', 'Count': cm_data.get('false_negatives', 0)}
                ])
                
                st.dataframe(cm_df, use_container_width=True, hide_index=True)
            
            # Clinical interpretation
            if evaluation:
                interpretation = st.session_state.evaluator.clinical_interpretation(evaluation)
                
                st.subheader("🏥 Clinical Interpretation")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Overall Performance:**")
                    st.info(interpretation['overall_performance'])
                    
                    st.write("**Clinical Utility:**")
                    for utility in interpretation['clinical_utility']:
                        st.write(f"• {utility}")
                
                with col2:
                    st.write("**Recommendations:**")
                    for rec in interpretation['recommendations']:
                        st.write(f"• {rec}")
        
        # Feature importance (for tree-based models)
        st.header("🎯 Feature Importance")
        
        tree_models = ['Decision_Tree', 'Random_Forest', 'Gradient_Boosting']
        available_tree_models = [model for model in tree_models if model in results]
        
        if available_tree_models:
            selected_tree_model = st.selectbox(
                "Select tree-based model for feature importance:",
                available_tree_models
            )
            
            if selected_tree_model:
                importance = st.session_state.ml_models.get_feature_importance(selected_tree_model)
                
                if importance is not None:
                    # Get feature names
                    if hasattr(st.session_state.data_processor, 'processed_features'):
                        feature_names = st.session_state.data_processor.processed_features
                    else:
                        feature_names = [f"Feature_{i}" for i in range(len(importance))]
                    
                    # Create feature importance plot
                    importance_fig = st.session_state.visualizer.plot_feature_importance(
                        importance, feature_names, selected_tree_model
                    )
                    
                    if importance_fig:
                        st.plotly_chart(importance_fig, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model.")
        else:
            st.info("No tree-based models trained. Feature importance is only available for Decision Tree, Random Forest, and Gradient Boosting models.")
        
        # Model export
        st.header("💾 Model Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save All Models"):
                try:
                    # Create models directory if it doesn't exist
                    os.makedirs('saved_models', exist_ok=True)
                    
                    # Save models
                    model_path = 'saved_models/ptsd_models.joblib'
                    st.session_state.ml_models.save_models(model_path)
                    
                    st.success(f"✅ Models saved to {model_path}")
                    
                except Exception as e:
                    st.error(f"Error saving models: {str(e)}")
        
        with col2:
            # Generate evaluation report
            if st.button("📄 Generate Report"):
                try:
                    report = st.session_state.evaluator.export_evaluation_report(selected_model)
                    
                    if report:
                        st.download_button(
                            label="📥 Download Report",
                            data=report,
                            file_name=f"ptsd_model_report_{selected_model}.md",
                            mime="text/markdown"
                        )
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")

if __name__ == "__main__":
    main()