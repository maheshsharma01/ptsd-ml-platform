import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.evaluation import ModelEvaluator
from utils.visualization import Visualizer
from database.database_manager import DatabaseManager

# Configure page
st.set_page_config(
    page_title="Model Comparison - PTSD ML Platform",
    page_icon="⚖️",
    layout="wide"
)

def main():
    st.title("⚖️ Model Comparison & Analysis")
    st.markdown("Compare different machine learning models and analyze their performance for PTSD prediction.")
    
    # Initialize components
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = ModelEvaluator()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer()
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    # Check if models are available
    if 'model_results' not in st.session_state or not st.session_state.model_results:
        st.warning("⚠️ No trained models available. Please train models first in the Model Training page.")
        
        # Show historical results from database
        st.subheader("📚 Historical Model Results")
        try:
            historical_results = st.session_state.db_manager.get_model_results()
            if not historical_results.empty:
                st.dataframe(historical_results, use_container_width=True)
            else:
                st.info("No historical model results found in database.")
        except Exception as e:
            st.error(f"Error loading historical results: {str(e)}")
        
        return
    
    results = st.session_state.model_results
    
    # Model selection for comparison
    st.header("🔍 Model Selection")
    
    available_models = list(results.keys())
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_models = st.multiselect(
            "Select models to compare:",
            available_models,
            default=available_models,
            help="Choose which models to include in the comparison"
        )
    
    with col2:
        comparison_metric = st.selectbox(
            "Primary comparison metric:",
            ["Accuracy", "F1-Score", "AUC", "Precision", "Recall"],
            help="Metric to use for ranking models"
        )
    
    if not selected_models:
        st.warning("Please select at least one model for comparison.")
        return
    
    # Filter results based on selection
    filtered_results = {name: results[name] for name in selected_models}
    
    # Comprehensive comparison table
    st.header("📊 Comprehensive Model Comparison")
    
    comparison_data = []
    for model_name, result in filtered_results.items():
        row = {
            'Model': model_name,
            'Accuracy': f"{result['accuracy']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1-Score': f"{result['f1_score']:.4f}",
            'AUC': f"{result.get('auc', 0):.4f}" if result.get('auc') else 'N/A',
            'CV Mean': f"{result['cv_mean']:.4f}",
            'CV Std': f"{result['cv_std']:.4f}",
            'Training Time': 'N/A'  # Would need to track this during training
        }
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by selected metric
    metric_map = {
        'Accuracy': 'Accuracy',
        'F1-Score': 'F1-Score',
        'AUC': 'AUC',
        'Precision': 'Precision',
        'Recall': 'Recall'
    }
    
    if metric_map[comparison_metric] in comparison_df.columns:
        # Convert to numeric for sorting
        sort_column = metric_map[comparison_metric]
        comparison_df[sort_column + '_numeric'] = pd.to_numeric(comparison_df[sort_column], errors='coerce')
        comparison_df = comparison_df.sort_values(sort_column + '_numeric', ascending=False)
        comparison_df = comparison_df.drop(sort_column + '_numeric', axis=1)
    
    # Add ranking
    comparison_df.insert(0, 'Rank', range(1, len(comparison_df) + 1))
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Best model highlight
    best_model = comparison_df.iloc[0]['Model']
    best_metric_value = comparison_df.iloc[0][metric_map[comparison_metric]]
    
    st.success(f"🏆 **Best Model by {comparison_metric}:** {best_model} ({best_metric_value})")
    
    # Performance visualizations
    st.header("📈 Performance Visualizations")
    
    # Create visualization tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Metrics Radar", "📈 Performance Trends", "🎯 ROC Analysis", 
        "⚖️ Precision-Recall", "🔄 Cross-Validation"
    ])
    
    with tab1:
        # Radar chart for multiple metrics
        st.subheader("Multi-Metric Radar Chart")
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        if any(result.get('auc') for result in filtered_results.values()):
            metrics.append('auc')
        
        fig_radar = go.Figure()
        
        for model_name, result in filtered_results.items():
            values = [result.get(metric, 0) for metric in metrics]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=[metric.replace('_', ' ').title() for metric in metrics],
                fill='toself',
                name=model_name
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab2:
        # Performance trends (bar charts)
        st.subheader("Performance Metrics Comparison")
        
        # Prepare data for plotting
        plot_data = []
        for model_name, result in filtered_results.items():
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                plot_data.append({
                    'Model': model_name,
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': result.get(metric, 0)
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        fig_bar = px.bar(
            plot_df,
            x='Model',
            y='Score',
            color='Metric',
            barmode='group',
            title='Performance Metrics Comparison',
            range_y=[0, 1]
        )
        
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab3:
        # ROC curves comparison
        st.subheader("ROC Curves Comparison")
        
        fig_roc = go.Figure()
        
        for model_name, result in filtered_results.items():
            if result.get('probabilities') is not None:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(result['actual'], result['probabilities'])
                auc_score = result.get('auc', 0)
                
                fig_roc.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'{model_name} (AUC = {auc_score:.3f})',
                    line=dict(width=3)
                ))
        
        # Add diagonal reference line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray', width=2),
            name='Random Classifier',
            showlegend=True
        ))
        
        fig_roc.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=700,
            height=500
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)
    
    with tab4:
        # Precision-Recall curves
        st.subheader("Precision-Recall Curves")
        
        fig_pr = go.Figure()
        
        for model_name, result in filtered_results.items():
            if result.get('probabilities') is not None:
                from sklearn.metrics import precision_recall_curve, average_precision_score
                precision, recall, _ = precision_recall_curve(result['actual'], result['probabilities'])
                ap_score = average_precision_score(result['actual'], result['probabilities'])
                
                fig_pr.add_trace(go.Scatter(
                    x=recall,
                    y=precision,
                    mode='lines',
                    name=f'{model_name} (AP = {ap_score:.3f})',
                    line=dict(width=3)
                ))
        
        fig_pr.update_layout(
            title='Precision-Recall Curves',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=700,
            height=500
        )
        
        st.plotly_chart(fig_pr, use_container_width=True)
    
    with tab5:
        # Cross-validation analysis
        st.subheader("Cross-Validation Performance")
        
        cv_data = []
        for model_name, result in filtered_results.items():
            cv_data.append({
                'Model': model_name,
                'CV Mean': result['cv_mean'],
                'CV Std': result['cv_std']
            })
        
        cv_df = pd.DataFrame(cv_data)
        
        fig_cv = px.bar(
            cv_df,
            x='Model',
            y='CV Mean',
            error_y='CV Std',
            title='Cross-Validation Scores with Standard Deviation',
            range_y=[0, 1]
        )
        
        fig_cv.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_cv, use_container_width=True)
    
    # Clinical interpretation comparison
    st.header("🏥 Clinical Interpretation Comparison")
    
    if 'evaluation_results' in st.session_state:
        evaluation_results = st.session_state.evaluation_results
        
        clinical_comparison = []
        
        for model_name in selected_models:
            if model_name in evaluation_results:
                evaluation = evaluation_results[model_name]
                clinical_metrics = evaluation.get('clinical_metrics', {})
                
                clinical_comparison.append({
                    'Model': model_name,
                    'Sensitivity': f"{clinical_metrics.get('sensitivity', 0):.3f}",
                    'Specificity': f"{clinical_metrics.get('specificity', 0):.3f}",
                    'PPV': f"{clinical_metrics.get('positive_predictive_value', 0):.3f}",
                    'NPV': f"{clinical_metrics.get('negative_predictive_value', 0):.3f}",
                    'LR+': f"{clinical_metrics.get('likelihood_ratio_positive', 0):.3f}",
                    'LR-': f"{clinical_metrics.get('likelihood_ratio_negative', 0):.3f}"
                })
        
        if clinical_comparison:
            clinical_df = pd.DataFrame(clinical_comparison)
            st.subheader("Clinical Metrics Comparison")
            st.dataframe(clinical_df, use_container_width=True, hide_index=True)
            
            # Clinical interpretation for each model
            st.subheader("Clinical Recommendations")
            
            for model_name in selected_models:
                if model_name in evaluation_results:
                    evaluation = evaluation_results[model_name]
                    interpretation = st.session_state.evaluator.clinical_interpretation(evaluation)
                    
                    with st.expander(f"Clinical Analysis: {model_name}"):
                        st.write(f"**Overall Performance:** {interpretation['overall_performance']}")
                        
                        st.write("**Clinical Utility:**")
                        for utility in interpretation['clinical_utility']:
                            st.write(f"• {utility}")
                        
                        st.write("**Recommendations:**")
                        for rec in interpretation['recommendations']:
                            st.write(f"• {rec}")
    
    # Model selection recommendations
    st.header("🎯 Model Selection Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("For Clinical Screening")
        st.markdown("""
        **Recommended Model:** Model with highest **Sensitivity**
        
        **Rationale:** In clinical screening, it's crucial to identify all potential PTSD cases 
        to avoid missing patients who need treatment. High sensitivity minimizes false negatives.
        """)
        
        # Find model with highest sensitivity
        if 'evaluation_results' in st.session_state:
            sensitivities = {}
            for model_name in selected_models:
                if model_name in st.session_state.evaluation_results:
                    clinical_metrics = st.session_state.evaluation_results[model_name].get('clinical_metrics', {})
                    sensitivities[model_name] = clinical_metrics.get('sensitivity', 0)
            
            if sensitivities:
                best_screening_model = max(sensitivities, key=sensitivities.get)
                st.success(f"**Recommended:** {best_screening_model} (Sensitivity: {sensitivities[best_screening_model]:.3f})")
    
    with col2:
        st.subheader("For Diagnostic Confirmation")
        st.markdown("""
        **Recommended Model:** Model with highest **Specificity** or **PPV**
        
        **Rationale:** For diagnostic confirmation, accuracy in positive predictions is critical 
        to avoid false positives that could lead to unnecessary treatment.
        """)
        
        # Find model with highest specificity
        if 'evaluation_results' in st.session_state:
            specificities = {}
            for model_name in selected_models:
                if model_name in st.session_state.evaluation_results:
                    clinical_metrics = st.session_state.evaluation_results[model_name].get('clinical_metrics', {})
                    specificities[model_name] = clinical_metrics.get('specificity', 0)
            
            if specificities:
                best_diagnostic_model = max(specificities, key=specificities.get)
                st.success(f"**Recommended:** {best_diagnostic_model} (Specificity: {specificities[best_diagnostic_model]:.3f})")
    
    # Export comparison report
    st.header("📄 Export Comparison Report")
    
    if st.button("📥 Generate Comparison Report"):
        # Create comprehensive report
        report = f"""
# PTSD Prediction Models Comparison Report

## Executive Summary
This report compares {len(selected_models)} machine learning models for PTSD prediction.

## Models Compared
{', '.join(selected_models)}

## Performance Summary
"""
        
        for idx, row in comparison_df.iterrows():
            report += f"\n{row['Rank']}. **{row['Model']}**\n"
            report += f"   - Accuracy: {row['Accuracy']}\n"
            report += f"   - F1-Score: {row['F1-Score']}\n"
            report += f"   - AUC: {row['AUC']}\n"
        
        report += f"\n## Best Model\n**{best_model}** achieved the highest {comparison_metric} of {best_metric_value}.\n"
        
        # Add clinical recommendations
        report += "\n## Clinical Recommendations\n"
        if 'evaluation_results' in st.session_state and sensitivities:
            report += f"- **For Screening:** {best_screening_model} (highest sensitivity)\n"
            report += f"- **For Diagnosis:** {best_diagnostic_model} (highest specificity)\n"
        
        report += "\n## Generated by PTSD ML Platform\n"
        
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="ptsd_models_comparison_report.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()