import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from database.database_manager import DatabaseManager

# Configure page
st.set_page_config(
    page_title="PTSD Prediction - PTSD ML Platform",
    page_icon="🔮",
    layout="wide"
)

def main():
    st.title("🔮 PTSD Prediction")
    st.markdown("Make predictions for individual patients or batch predictions for multiple cases.")
    
    # Initialize database manager
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    # Check if models are trained
    if 'models_trained' not in st.session_state or not st.session_state.models_trained:
        st.warning("⚠️ No trained models available. Please train models first in the Model Training page.")
        return
    
    if 'ml_models' not in st.session_state:
        st.warning("⚠️ ML models not initialized. Please train models first.")
        return
    
    # Prediction type selection
    st.header("🎯 Prediction Type")
    
    prediction_type = st.radio(
        "Choose prediction type:",
        ["Individual Prediction", "Batch Prediction"],
        horizontal=True
    )
    
    if prediction_type == "Individual Prediction":
        individual_prediction()
    else:
        batch_prediction()

def individual_prediction():
    """Handle individual patient prediction."""
    
    st.subheader("👤 Individual Patient Prediction")
    
    # Model selection
    if 'model_results' in st.session_state:
        available_models = list(st.session_state.model_results.keys())
        selected_model = st.selectbox(
            "Select prediction model:",
            available_models,
            index=0 if available_models else None
        )
    else:
        st.error("No trained models available.")
        return
    
    # Input form
    st.subheader("📝 Patient Information")
    
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Demographic Information**")
            patient_id = st.text_input("Patient ID", value="", help="Unique patient identifier")
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            trauma_type = st.selectbox(
                "Trauma Type",
                ["Combat", "Sexual Assault", "Motor Vehicle Accident", "Natural Disaster", "Physical Assault", "Other"]
            )
        
        with col2:
            st.write("**Clinical Information**")
            time_since_trauma = st.number_input("Time Since Trauma (months)", min_value=0, max_value=600, value=6)
            education_years = st.number_input("Education (years)", min_value=0, max_value=25, value=12)
            employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Student", "Retired"])
            previous_trauma = st.checkbox("Previous Trauma History")
            family_history = st.checkbox("Family History of Mental Health Issues")
        
        st.write("**PCL-5 Scale Scores**")
        col3, col4 = st.columns(2)
        
        with col3:
            pcl5_intrusive = st.slider("Intrusive Memories", 0.0, 5.0, 2.5, 0.1)
            pcl5_avoidance = st.slider("Avoidance", 0.0, 5.0, 2.5, 0.1)
        
        with col4:
            pcl5_negative_mood = st.slider("Negative Mood", 0.0, 5.0, 2.5, 0.1)
            pcl5_hyperarousal = st.slider("Hyperarousal", 0.0, 5.0, 2.5, 0.1)
        
        # Calculate total PCL-5 score
        pcl5_total = pcl5_intrusive + pcl5_avoidance + pcl5_negative_mood + pcl5_hyperarousal
        st.info(f"**PCL-5 Total Score:** {pcl5_total:.1f}/20")
        
        st.write("**Biomarkers**")
        cortisol_level = st.number_input("Cortisol Level (μg/dL)", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
        social_support = st.slider("Social Support Score", 0.0, 10.0, 5.0, 0.1)
        
        # Submit button
        submitted = st.form_submit_button("🔮 Make Prediction", type="primary")
        
        if submitted:
            # Prepare input data
            input_data = {
                'age': age,
                'gender': 1 if gender == 'Female' else 0,  # Simple encoding
                'trauma_type': hash(trauma_type) % 10,  # Simple hash encoding
                'time_since_trauma': time_since_trauma,
                'education_years': education_years,
                'employment_status': 1 if employment_status == 'Employed' else 0,
                'previous_trauma': int(previous_trauma),
                'family_history_mental_health': int(family_history),
                'pcl5_intrusive': pcl5_intrusive,
                'pcl5_avoidance': pcl5_avoidance,
                'pcl5_negative_mood': pcl5_negative_mood,
                'pcl5_hyperarousal': pcl5_hyperarousal,
                'pcl5_total_score': pcl5_total,
                'cortisol_level': cortisol_level,
                'social_support_score': social_support
            }
            
            # Convert to DataFrame for processing
            input_df = pd.DataFrame([input_data])
            
            try:
                # Process input using the same pipeline
                if hasattr(st.session_state.data_processor, 'processed_features'):
                    # Ensure all required features are present
                    for feature in st.session_state.data_processor.processed_features:
                        if feature not in input_df.columns:
                            input_df[feature] = 0  # Default value for missing features
                    
                    # Select only the features used in training
                    input_df = input_df[st.session_state.data_processor.processed_features]
                
                # Transform the input data
                X_input = st.session_state.data_processor.transform_new_data(input_df)
                
                # Make prediction
                prediction, probability = st.session_state.ml_models.predict(X_input, selected_model)
                
                # Determine risk level
                if probability is not None:
                    prob_value = probability[0]
                    if prob_value < 0.3:
                        risk_level = "Low"
                        risk_color = "green"
                    elif prob_value < 0.7:
                        risk_level = "Moderate"
                        risk_color = "orange"
                    else:
                        risk_level = "High"
                        risk_color = "red"
                else:
                    prob_value = None
                    risk_level = "Unknown"
                    risk_color = "gray"
                
                # Display results
                st.success("✅ Prediction completed!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    prediction_text = "PTSD Positive" if prediction[0] == 1 else "PTSD Negative"
                    st.metric("Prediction", prediction_text)
                
                with col2:
                    if prob_value is not None:
                        st.metric("Probability", f"{prob_value:.3f}")
                    else:
                        st.metric("Probability", "N/A")
                
                with col3:
                    st.metric("Risk Level", risk_level)
                
                # Visualization
                if prob_value is not None:
                    # Create probability gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prob_value,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "PTSD Probability"},
                        delta={'reference': 0.5},
                        gauge={
                            'axis': {'range': [None, 1]},
                            'bar': {'color': risk_color},
                            'steps': [
                                {'range': [0, 0.3], 'color': "lightgreen"},
                                {'range': [0.3, 0.7], 'color': "lightyellow"},
                                {'range': [0.7, 1], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.5
                            }
                        }
                    ))
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Clinical interpretation
                st.subheader("🏥 Clinical Interpretation")
                
                if prediction[0] == 1:
                    st.error("**Positive PTSD Prediction** - Patient shows indicators consistent with PTSD diagnosis.")
                    if prob_value and prob_value > 0.8:
                        st.write("• High confidence prediction - Consider comprehensive clinical assessment")
                    elif prob_value and prob_value > 0.6:
                        st.write("• Moderate confidence prediction - Additional evaluation recommended")
                    else:
                        st.write("• Lower confidence prediction - Clinical judgment advised")
                else:
                    st.success("**Negative PTSD Prediction** - Patient shows lower likelihood of PTSD diagnosis.")
                    if prob_value and prob_value < 0.2:
                        st.write("• High confidence prediction - Low PTSD likelihood")
                    elif prob_value and prob_value < 0.4:
                        st.write("• Moderate confidence prediction - Monitor for symptoms")
                    else:
                        st.write("• Borderline prediction - Consider follow-up assessment")
                
                # Feature contributions (if available)
                if selected_model in ['Decision_Tree', 'Random_Forest', 'Gradient_Boosting']:
                    importance = st.session_state.ml_models.get_feature_importance(selected_model)
                    if importance is not None and hasattr(st.session_state.data_processor, 'processed_features'):
                        st.subheader("📊 Key Contributing Factors")
                        
                        # Get top 5 most important features
                        feature_names = st.session_state.data_processor.processed_features
                        top_indices = np.argsort(importance)[-5:][::-1]
                        
                        for idx in top_indices:
                            if idx < len(feature_names):
                                feature_name = feature_names[idx]
                                feature_importance = importance[idx]
                                st.write(f"• **{feature_name}**: {feature_importance:.3f} importance")
                
                # Save prediction to database
                if patient_id:
                    success = st.session_state.db_manager.save_prediction(
                        patient_id=patient_id,
                        model_name=selected_model,
                        prediction=int(prediction[0]),
                        probability=float(prob_value) if prob_value is not None else None,
                        feature_values=input_data,
                        risk_level=risk_level,
                        notes=f"Prediction made using {selected_model} model"
                    )
                    
                    if success:
                        st.info("💾 Prediction saved to database")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.write("Please ensure all required features are provided and models are properly trained.")

def batch_prediction():
    """Handle batch prediction for multiple patients."""
    
    st.subheader("📊 Batch Prediction")
    
    # Model selection
    if 'model_results' in st.session_state:
        available_models = list(st.session_state.model_results.keys())
        selected_model = st.selectbox(
            "Select prediction model:",
            available_models,
            index=0 if available_models else None,
            key="batch_model"
        )
    else:
        st.error("No trained models available.")
        return
    
    # File upload for batch prediction
    uploaded_file = st.file_uploader(
        "Upload file for batch prediction",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file with patient data for batch prediction"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                batch_df = pd.read_csv(uploaded_file)
            else:
                batch_df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(batch_df)} records for prediction")
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(batch_df.head())
            
            # Make predictions button
            if st.button("🚀 Make Batch Predictions", type="primary"):
                with st.spinner("Making predictions..."):
                    try:
                        # Process the batch data
                        X_batch = st.session_state.data_processor.transform_new_data(batch_df)
                        
                        # Make predictions
                        predictions, probabilities = st.session_state.ml_models.predict(X_batch, selected_model)
                        
                        # Add predictions to dataframe
                        results_df = batch_df.copy()
                        results_df['PTSD_Prediction'] = predictions
                        results_df['PTSD_Prediction_Label'] = ['PTSD Positive' if p == 1 else 'PTSD Negative' for p in predictions]
                        
                        if probabilities is not None:
                            results_df['PTSD_Probability'] = probabilities
                            results_df['Risk_Level'] = pd.cut(
                                probabilities,
                                bins=[0, 0.3, 0.7, 1.0],
                                labels=['Low', 'Moderate', 'High']
                            )
                        
                        # Display results
                        st.success("✅ Batch predictions completed!")
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_cases = len(predictions)
                            st.metric("Total Cases", total_cases)
                        
                        with col2:
                            positive_cases = sum(predictions)
                            st.metric("PTSD Positive", positive_cases)
                        
                        with col3:
                            negative_cases = total_cases - positive_cases
                            st.metric("PTSD Negative", negative_cases)
                        
                        with col4:
                            if total_cases > 0:
                                positive_rate = (positive_cases / total_cases) * 100
                                st.metric("Positive Rate", f"{positive_rate:.1f}%")
                        
                        # Results visualization
                        st.subheader("📈 Results Visualization")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Prediction distribution
                            pred_counts = pd.Series(predictions).value_counts()
                            fig_pred = px.pie(
                                values=pred_counts.values,
                                names=['PTSD Negative', 'PTSD Positive'],
                                title='Prediction Distribution'
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)
                        
                        with col2:
                            if probabilities is not None:
                                # Probability distribution
                                fig_prob = px.histogram(
                                    x=probabilities,
                                    title='Probability Distribution',
                                    nbins=20,
                                    labels={'x': 'PTSD Probability', 'y': 'Count'}
                                )
                                st.plotly_chart(fig_prob, use_container_width=True)
                        
                        # Results table
                        st.subheader("📋 Detailed Results")
                        
                        # Display options
                        show_all = st.checkbox("Show all columns", value=False)
                        
                        if show_all:
                            display_df = results_df
                        else:
                            # Show only key columns
                            key_columns = ['patient_id'] if 'patient_id' in results_df.columns else []
                            key_columns.extend(['PTSD_Prediction_Label', 'PTSD_Probability', 'Risk_Level'])
                            key_columns = [col for col in key_columns if col in results_df.columns]
                            display_df = results_df[key_columns] if key_columns else results_df
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"ptsd_predictions_{selected_model}.csv",
                            mime="text/csv"
                        )
                        
                        # Save to database option
                        if st.button("💾 Save Results to Database"):
                            with st.spinner("Saving to database..."):
                                saved_count = 0
                                
                                for idx, row in results_df.iterrows():
                                    patient_id = row.get('patient_id', f'BATCH_{idx}')
                                    prediction = int(row['PTSD_Prediction'])
                                    probability = float(row['PTSD_Probability']) if 'PTSD_Probability' in row else None
                                    risk_level = row.get('Risk_Level', 'Unknown')
                                    
                                    # Prepare feature values
                                    feature_values = row.drop(['PTSD_Prediction', 'PTSD_Prediction_Label', 'PTSD_Probability', 'Risk_Level'], errors='ignore').to_dict()
                                    
                                    success = st.session_state.db_manager.save_prediction(
                                        patient_id=patient_id,
                                        model_name=selected_model,
                                        prediction=prediction,
                                        probability=probability,
                                        feature_values=feature_values,
                                        risk_level=risk_level,
                                        notes="Batch prediction"
                                    )
                                    
                                    if success:
                                        saved_count += 1
                                
                                st.success(f"✅ Saved {saved_count}/{len(results_df)} predictions to database")
                        
                    except Exception as e:
                        st.error(f"Error during batch prediction: {str(e)}")
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    else:
        # Show sample format
        st.subheader("📝 Sample Batch File Format")
        st.markdown("""
        Upload a CSV or Excel file with the following columns:
        
        **Required columns:**
        - patient_id (optional but recommended)
        - age, gender, trauma_type
        - pcl5_intrusive, pcl5_avoidance, pcl5_negative_mood, pcl5_hyperarousal
        - cortisol_level, social_support_score
        - time_since_trauma, education_years, employment_status
        - previous_trauma, family_history_mental_health
        """)
        
        # Sample data for download
        sample_batch = pd.DataFrame({
            'patient_id': ['B001', 'B002', 'B003'],
            'age': [28, 35, 42],
            'gender': ['Female', 'Male', 'Female'],
            'trauma_type': ['Combat', 'Accident', 'Assault'],
            'pcl5_intrusive': [3.5, 2.2, 4.1],
            'pcl5_avoidance': [3.0, 2.8, 3.9],
            'pcl5_negative_mood': [3.8, 1.9, 4.2],
            'pcl5_hyperarousal': [3.2, 2.5, 3.8],
            'cortisol_level': [16.5, 11.2, 19.8],
            'social_support_score': [4.2, 6.8, 3.1],
            'time_since_trauma': [12, 6, 18],
            'education_years': [14, 16, 12],
            'employment_status': ['Employed', 'Unemployed', 'Employed'],
            'previous_trauma': [True, False, True],
            'family_history_mental_health': [False, True, True]
        })
        
        st.dataframe(sample_batch)
        
        csv = sample_batch.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample File",
            data=csv,
            file_name="batch_prediction_template.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()