import streamlit as st
import pandas as pd
import numpy as np
from models.ml_models import PTSDMLModels
from data.data_processor import DataProcessor
from utils.visualization import Visualizer
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="PTSD Prediction Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🧠 PTSD Prediction Platform")
    st.markdown("""
    ### Machine Learning-Based Post-Traumatic Stress Disorder Prediction
    
    This platform implements multiple machine learning algorithms for PTSD prediction based on comprehensive research.
    It supports various data types including psychometric scales, biomarkers, and neuroimaging features.
    """)
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'ml_models' not in st.session_state:
        st.session_state.ml_models = PTSDMLModels()
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    # Main dashboard overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Status", "Ready" if st.session_state.data_loaded else "Not Loaded")
    
    with col2:
        st.metric("Models Trained", "Yes" if st.session_state.models_trained else "No")
    
    with col3:
        if st.session_state.data_loaded and hasattr(st.session_state, 'processed_data'):
            n_samples = len(st.session_state.processed_data)
            st.metric("Total Samples", n_samples)
        else:
            st.metric("Total Samples", "0")
    
    with col4:
        if st.session_state.models_trained and hasattr(st.session_state, 'model_results'):
            best_accuracy = max([results['accuracy'] for results in st.session_state.model_results.values()])
            st.metric("Best Model Accuracy", f"{best_accuracy:.3f}")
        else:
            st.metric("Best Model Accuracy", "N/A")
    
    st.markdown("---")
    
    # Quick start section
    st.header("🚀 Quick Start Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Input Options")
        st.markdown("""
        **Supported Data Types:**
        - DSM-5 PCL-5 Psychometric Scales
        - Biomarker Data (Cortisol levels, etc.)
        - Neuroimaging Features (MEG, sMRI)
        - Clinical Interview Data
        - Demographic Information
        """)
    
    with col2:
        st.subheader("🤖 Available ML Models")
        st.markdown("""
        **Implemented Algorithms:**
        - Support Vector Machines (SVM)
        - Artificial Neural Networks (ANN)
        - Decision Trees (DT)
        - Gaussian Naive Bayes (GNB)
        - Random Forest (RF)
        - Gradient Boosting (GB)
        """)
    
    # Recent research highlights
    st.header("📈 Research Highlights")
    
    research_metrics = {
        "SVM with Neuroimaging": {"accuracy": "82-90%", "data_type": "MEG/sMRI"},
        "ANN with PCL-5": {"accuracy": "90%", "data_type": "Psychometric"},
        "Cortisol + Clinical": {"auc": "0.82", "data_type": "Biomarkers"},
        "NLP Models": {"auc": "0.72", "data_type": "Language"}
    }
    
    cols = st.columns(len(research_metrics))
    for i, (model, metrics) in enumerate(research_metrics.items()):
        with cols[i]:
            st.info(f"**{model}**\n\n{metrics.get('accuracy', metrics.get('auc', 'N/A'))}\n\n*{metrics['data_type']}*")
    
    # Navigation instructions
    st.header("📋 Getting Started")
    st.markdown("""
    1. **Upload Data**: Go to the 'Data Upload' page to load your dataset
    2. **Train Models**: Use the 'Model Training' page to train ML algorithms
    3. **Make Predictions**: Use the 'Prediction' page for individual or batch predictions
    4. **Compare Models**: Analyze different algorithms in the 'Model Comparison' page
    """)
    
    # Sample data section
    st.header("📝 Sample Data Template")
    st.markdown("Download a template to see the expected data format:")
    
    # Create sample PCL-5 template
    sample_data = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003'],
        'age': [25, 34, 45],
        'gender': ['F', 'M', 'F'],
        'trauma_type': ['Combat', 'Accident', 'Natural Disaster'],
        'pcl5_intrusive': [3.2, 2.8, 4.1],
        'pcl5_avoidance': [2.9, 3.5, 3.8],
        'pcl5_negative_mood': [3.4, 2.1, 4.2],
        'pcl5_hyperarousal': [3.1, 2.6, 3.9],
        'cortisol_level': [15.2, 12.8, 18.5],
        'ptsd_diagnosis': [1, 0, 1]  # 1 = PTSD, 0 = No PTSD
    })
    
    st.dataframe(sample_data)
    
    # Convert to CSV for download
    csv = sample_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample Template",
        data=csv,
        file_name="ptsd_data_template.csv",
        mime="text/csv"
    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>PTSD Prediction Platform</strong> | Based on comprehensive machine learning research</p>
        <p><em>Implementing SVM, ANN, Decision Trees, and other ML approaches for clinical PTSD prediction</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()