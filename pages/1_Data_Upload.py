import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data.data_processor import DataProcessor
from utils.visualization import Visualizer
from database.database_manager import DatabaseManager
import io

# Configure page
st.set_page_config(
    page_title="Data Upload - PTSD ML Platform",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("📊 Data Upload & Processing")
    st.markdown("Upload your PTSD prediction dataset and perform initial analysis.")
    
    # Initialize components
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer()
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    # Sidebar for options
    st.sidebar.header("Data Upload Options")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel files with PTSD prediction data"
    )
    
    if uploaded_file is not None:
        # Load data
        with st.spinner("Loading data..."):
            df = st.session_state.data_processor.load_data(uploaded_file)
        
        if df is not None:
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            
            # Store in session state
            st.session_state.uploaded_data = df
            st.session_state.data_loaded = True
            
            # Data overview
            st.header("📋 Data Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                missing_percent = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                st.metric("Missing %", f"{missing_percent:.1f}%")
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Data analysis
            st.header("🔍 Data Analysis")
            
            with st.spinner("Analyzing data..."):
                analysis = st.session_state.data_processor.analyze_data(df)
            
            # Display analysis results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Column Information")
                
                # Create data info table
                data_info = pd.DataFrame({
                    'Column': analysis['columns'],
                    'Type': [str(analysis['dtypes'][col]) for col in analysis['columns']],
                    'Missing': [analysis['missing_values'][col] for col in analysis['columns']],
                    'Missing %': [(analysis['missing_values'][col] / len(df)) * 100 for col in analysis['columns']]
                })
                
                # Color code missing values
                def color_missing(val):
                    if val > 20:
                        return 'background-color: #ffcccc'
                    elif val > 10:
                        return 'background-color: #ffffcc'
                    else:
                        return 'background-color: #ccffcc'
                
                styled_df = data_info.style.applymap(color_missing, subset=['Missing %'])
                st.dataframe(styled_df, use_container_width=True)
            
            with col2:
                st.subheader("Data Categories")
                
                st.write("**Numeric Columns:**")
                if analysis['numeric_columns']:
                    for col in analysis['numeric_columns']:
                        st.write(f"• {col}")
                else:
                    st.write("None found")
                
                st.write("**Categorical Columns:**")
                if analysis['categorical_columns']:
                    for col in analysis['categorical_columns']:
                        st.write(f"• {col}")
                else:
                    st.write("None found")
                
                st.write("**Potential Target Columns:**")
                if analysis['target_candidates']:
                    for col in analysis['target_candidates']:
                        st.write(f"• {col}")
                else:
                    st.write("None automatically detected")
            
            # Specialized column detection
            st.subheader("Specialized Data Detection")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**PCL-5 Columns:**")
                if analysis['pcl5_columns']:
                    for col in analysis['pcl5_columns']:
                        st.write(f"• {col}")
                else:
                    st.write("None detected")
            
            with col2:
                st.write("**Biomarker Columns:**")
                if analysis['biomarker_columns']:
                    for col in analysis['biomarker_columns']:
                        st.write(f"• {col}")
                else:
                    st.write("None detected")
            
            with col3:
                st.write("**Neuroimaging Columns:**")
                if analysis['neuroimaging_columns']:
                    for col in analysis['neuroimaging_columns']:
                        st.write(f"• {col}")
                else:
                    st.write("None detected")
            
            # Data visualization
            st.header("📈 Data Visualization")
            
            # Select target column for visualization
            target_options = ['None'] + analysis['target_candidates'] + analysis['columns']
            selected_target = st.selectbox(
                "Select target column for analysis:",
                target_options,
                index=1 if analysis['target_candidates'] else 0
            )
            
            target_col = selected_target if selected_target != 'None' else None
            
            # Generate visualizations
            with st.spinner("Creating visualizations..."):
                plots = st.session_state.visualizer.plot_data_overview(df, target_col)
            
            # Display plots
            if 'missing_values' in plots:
                st.subheader("Missing Values Analysis")
                st.plotly_chart(plots['missing_values'], use_container_width=True)
            
            if 'distributions' in plots:
                st.subheader("Feature Distributions")
                st.plotly_chart(plots['distributions'], use_container_width=True)
            
            if 'target_distribution' in plots and target_col:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"Target Distribution: {target_col}")
                    st.plotly_chart(plots['target_distribution'], use_container_width=True)
                
                with col2:
                    if 'target_correlation' in plots:
                        st.subheader("Feature-Target Correlations")
                        st.plotly_chart(plots['target_correlation'], use_container_width=True)
            
            # PCL-5 specific analysis
            if analysis['pcl5_columns']:
                st.header("🧠 PCL-5 Analysis")
                
                # Create enhanced PCL-5 features
                with st.spinner("Analyzing PCL-5 data..."):
                    df_enhanced = st.session_state.data_processor.create_pcl5_features(df)
                    pcl5_plots = st.session_state.visualizer.plot_pcl5_analysis(df_enhanced)
                
                if pcl5_plots:
                    for plot_name, fig in pcl5_plots.items():
                        st.plotly_chart(fig, use_container_width=True)
            
            # Data quality validation
            st.header("✅ Data Quality Validation")
            
            validation = st.session_state.data_processor.validate_data_quality(df)
            
            if validation['issues']:
                st.subheader("⚠️ Issues Found")
                for issue in validation['issues']:
                    st.error(issue)
            
            if validation['warnings']:
                st.subheader("⚡ Warnings")
                for warning in validation['warnings']:
                    st.warning(warning)
            
            st.subheader("💡 Recommendations")
            for rec in validation['recommendations']:
                st.info(rec)
            
            # Data preprocessing options
            st.header("🔧 Data Preprocessing")
            
            with st.expander("Preprocessing Options", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Target column selection
                    target_column = st.selectbox(
                        "Select target column:",
                        options=[''] + analysis['columns'],
                        help="Choose the column containing PTSD diagnosis (0/1)"
                    )
                    
                    # Missing value handling
                    missing_strategy = st.selectbox(
                        "Missing value strategy:",
                        options=['mean', 'median', 'mode', 'drop'],
                        help="Strategy for handling missing values"
                    )
                
                with col2:
                    # Feature selection
                    encode_categorical = st.checkbox(
                        "Encode categorical variables",
                        value=True,
                        help="Convert categorical variables to numeric"
                    )
                    
                    scale_features = st.checkbox(
                        "Scale features",
                        value=True,
                        help="Standardize numeric features"
                    )
                
                if st.button("🚀 Process Data"):
                    if target_column:
                        with st.spinner("Processing data..."):
                            try:
                                X, y = st.session_state.data_processor.preprocess_data(
                                    df,
                                    target_column=target_column,
                                    handle_missing=missing_strategy,
                                    encode_categorical=encode_categorical,
                                    scale_features=scale_features
                                )
                                
                                # Store processed data
                                st.session_state.processed_data = df
                                st.session_state.features = X
                                st.session_state.target = y
                                st.session_state.target_column = target_column
                                st.session_state.data_processed = True
                                
                                st.success(f"✅ Data processed successfully! Features shape: {X.shape}")
                                
                                # Show processed data summary
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Processed Samples", X.shape[0])
                                with col2:
                                    st.metric("Features", X.shape[1])
                                with col3:
                                    class_dist = pd.Series(y).value_counts()
                                    balance = min(class_dist) / max(class_dist) if len(class_dist) > 1 else 1
                                    st.metric("Class Balance", f"{balance:.2f}")
                                
                            except Exception as e:
                                st.error(f"Error processing data: {str(e)}")
                    else:
                        st.warning("Please select a target column first.")
            
            # Database operations
            st.header("💾 Database Operations")
            
            with st.expander("Save to Database", expanded=False):
                st.write("Save processed data to the database for future use.")
                
                if st.button("💾 Save to Database"):
                    if 'processed_data' in st.session_state:
                        with st.spinner("Saving to database..."):
                            try:
                                # Convert DataFrame to patient records
                                success_count = 0
                                total_count = len(st.session_state.processed_data)
                                
                                for idx, row in st.session_state.processed_data.iterrows():
                                    patient_data = {
                                        'patient_id': f"P{idx:04d}",
                                        'age': row.get('age'),
                                        'gender': row.get('gender'),
                                        'trauma_type': row.get('trauma_type'),
                                        'pcl5_intrusive': row.get('pcl5_intrusive'),
                                        'pcl5_avoidance': row.get('pcl5_avoidance'),
                                        'pcl5_negative_mood': row.get('pcl5_negative_mood'),
                                        'pcl5_hyperarousal': row.get('pcl5_hyperarousal'),
                                        'pcl5_total_score': row.get('pcl5_total_score'),
                                        'cortisol_level': row.get('cortisol_level'),
                                        'ptsd_diagnosis': row.get(st.session_state.get('target_column', 'ptsd_diagnosis'))
                                    }
                                    
                                    # Remove None values
                                    patient_data = {k: v for k, v in patient_data.items() if v is not None}
                                    
                                    if st.session_state.db_manager.save_patient_data(patient_data):
                                        success_count += 1
                                
                                st.success(f"✅ Saved {success_count}/{total_count} records to database!")
                                
                            except Exception as e:
                                st.error(f"Error saving to database: {str(e)}")
                    else:
                        st.warning("No processed data available. Please process data first.")
    
    else:
        # Show sample data format
        st.header("📝 Sample Data Format")
        st.markdown("""
        Upload a CSV or Excel file with PTSD prediction data. Your data should include:
        
        **Required Columns:**
        - Patient identifier
        - PTSD diagnosis (0 = No PTSD, 1 = PTSD)
        
        **Recommended Columns:**
        - Demographic data (age, gender)
        - PCL-5 scale scores
        - Biomarker data (cortisol levels)
        - Clinical assessments
        """)
        
        # Show sample template
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
            'ptsd_diagnosis': [1, 0, 1]
        })
        
        st.subheader("Sample Data Template")
        st.dataframe(sample_data)
        
        # Download template
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Template",
            data=csv,
            file_name="ptsd_data_template.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()