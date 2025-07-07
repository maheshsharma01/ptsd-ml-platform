import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

class DataProcessor:
    """
    Data processing utilities for PTSD prediction datasets.
    Handles various data types including psychometric scales, biomarkers, and clinical data.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        self.feature_selector = None
        self.processed_features = None
        self.target_column = None
        
    def load_data(self, uploaded_file):
        """
        Load data from uploaded file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            DataFrame with loaded data
        """
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Unsupported file format. Please upload CSV or Excel files.")
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def analyze_data(self, df):
        """
        Analyze dataset and provide summary statistics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with data analysis results
        """
        analysis = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'target_candidates': []
        }
        
        # Identify potential target columns
        potential_targets = ['ptsd', 'diagnosis', 'ptsd_diagnosis', 'label', 'target', 'outcome']
        for col in df.columns:
            if any(target in col.lower() for target in potential_targets):
                analysis['target_candidates'].append(col)
        
        # Check for PCL-5 scale columns
        pcl5_columns = [col for col in df.columns if 'pcl5' in col.lower() or 'pcl-5' in col.lower()]
        analysis['pcl5_columns'] = pcl5_columns
        
        # Check for biomarker columns
        biomarker_keywords = ['cortisol', 'hormone', 'biomarker', 'blood', 'serum']
        biomarker_columns = [col for col in df.columns 
                           if any(keyword in col.lower() for keyword in biomarker_keywords)]
        analysis['biomarker_columns'] = biomarker_columns
        
        # Check for neuroimaging columns
        neuroimaging_keywords = ['meg', 'mri', 'eeg', 'brain', 'neural', 'connectivity']
        neuroimaging_columns = [col for col in df.columns 
                              if any(keyword in col.lower() for keyword in neuroimaging_keywords)]
        analysis['neuroimaging_columns'] = neuroimaging_columns
        
        return analysis
    
    def preprocess_data(self, df, target_column, feature_columns=None, 
                       handle_missing='mean', encode_categorical=True,
                       scale_features=True):
        """
        Comprehensive data preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            feature_columns: List of feature columns to use (if None, uses all except target)
            handle_missing: Strategy for missing values ('mean', 'median', 'mode', 'drop')
            encode_categorical: Whether to encode categorical variables
            scale_features: Whether to scale numerical features
            
        Returns:
            Processed features (X) and target (y)
        """
        
        df_processed = df.copy()
        
        # Validate target column
        if target_column not in df_processed.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Select feature columns
        if feature_columns is None:
            feature_columns = [col for col in df_processed.columns if col != target_column]
        
        # Extract target
        y = df_processed[target_column].values
        X = df_processed[feature_columns].copy()
        
        # Handle missing values in features
        if handle_missing != 'drop':
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            categorical_columns = X.select_dtypes(include=['object']).columns
            
            # Impute numeric columns
            if len(numeric_columns) > 0:
                strategy = 'mean' if handle_missing == 'mean' else 'median'
                numeric_imputer = SimpleImputer(strategy=strategy)
                X[numeric_columns] = numeric_imputer.fit_transform(X[numeric_columns])
                self.imputers['numeric'] = numeric_imputer
            
            # Impute categorical columns
            if len(categorical_columns) > 0:
                categorical_imputer = SimpleImputer(strategy='most_frequent')
                X[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns])
                self.imputers['categorical'] = categorical_imputer
        else:
            # Drop rows with missing values
            mask = ~(X.isnull().any(axis=1) | pd.isna(y))
            X = X[mask]
            y = y[mask]
        
        # Encode categorical variables
        if encode_categorical:
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Convert to numpy arrays
        X_array = X.values.astype(float)
        
        # Scale features
        if scale_features:
            X_array = self.scaler.fit_transform(X_array)
        
        # Store processed feature names
        self.processed_features = feature_columns
        self.target_column = target_column
        
        return X_array, y
    
    def feature_selection(self, X, y, method='univariate', k=10):
        """
        Perform feature selection.
        
        Args:
            X: Feature matrix
            y: Target vector
            method: Selection method ('univariate', 'rfe', 'importance')
            k: Number of features to select
            
        Returns:
            Selected features indices and transformed X
        """
        
        if method == 'univariate':
            # Univariate feature selection
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            
        elif method == 'rfe':
            # Recursive Feature Elimination
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=k)
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            
        elif method == 'importance':
            # Feature importance based selection
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:k]
            X_selected = X[:, indices]
            selected_indices = indices
            
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        self.feature_selector = selector if method in ['univariate', 'rfe'] else None
        
        return X_selected, selected_indices
    
    def create_pcl5_features(self, df):
        """
        Create derived features from PCL-5 scale data.
        
        Args:
            df: DataFrame with PCL-5 columns
            
        Returns:
            DataFrame with additional PCL-5 derived features
        """
        
        df_enhanced = df.copy()
        
        # PCL-5 subscale patterns based on research
        pcl5_subscales = {
            'intrusive': ['pcl5_1', 'pcl5_2', 'pcl5_3', 'pcl5_4', 'pcl5_5'],
            'avoidance': ['pcl5_6', 'pcl5_7'],
            'negative_mood': ['pcl5_8', 'pcl5_9', 'pcl5_10', 'pcl5_11', 'pcl5_12', 'pcl5_13', 'pcl5_14'],
            'hyperarousal': ['pcl5_15', 'pcl5_16', 'pcl5_17', 'pcl5_18', 'pcl5_19', 'pcl5_20']
        }
        
        # Check for existing PCL-5 columns and create subscale scores
        for subscale, items in pcl5_subscales.items():
            available_items = [item for item in items if item in df.columns]
            if available_items:
                df_enhanced[f'pcl5_{subscale}_score'] = df_enhanced[available_items].mean(axis=1)
        
        # Create total PCL-5 score if individual items exist
        pcl5_items = [col for col in df.columns if col.startswith('pcl5_') and col.split('_')[-1].isdigit()]
        if pcl5_items:
            df_enhanced['pcl5_total_score'] = df_enhanced[pcl5_items].sum(axis=1)
            df_enhanced['pcl5_mean_score'] = df_enhanced[pcl5_items].mean(axis=1)
        
        # Create severity categories based on research thresholds
        if 'pcl5_total_score' in df_enhanced.columns:
            df_enhanced['pcl5_severity'] = pd.cut(
                df_enhanced['pcl5_total_score'],
                bins=[0, 33, 45, 57, 80],
                labels=['Minimal', 'Mild', 'Moderate', 'Severe']
            )
        
        return df_enhanced
    
    def validate_data_quality(self, df):
        """
        Validate data quality and identify potential issues.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with validation results
        """
        
        validation = {
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check for excessive missing values
        missing_percent = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_percent[missing_percent > 20]
        if len(high_missing) > 0:
            validation['warnings'].append(f"Columns with >20% missing values: {high_missing.index.tolist()}")
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            validation['issues'].append(f"Found {duplicates} duplicate rows")
        
        # Check for constant columns
        constant_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        if constant_cols:
            validation['issues'].append(f"Constant columns found: {constant_cols}")
        
        # Check sample size
        if len(df) < 100:
            validation['warnings'].append("Small sample size (<100) may affect model performance")
        
        # Check class balance if target column is identified
        potential_targets = [col for col in df.columns 
                           if 'ptsd' in col.lower() or 'diagnosis' in col.lower()]
        for target_col in potential_targets:
            if target_col in df.columns:
                class_counts = df[target_col].value_counts()
                class_balance = min(class_counts) / max(class_counts)
                if class_balance < 0.3:
                    validation['warnings'].append(f"Class imbalance detected in {target_col}")
        
        # Recommendations
        if len(validation['issues']) == 0 and len(validation['warnings']) == 0:
            validation['recommendations'].append("Data quality looks good!")
        else:
            validation['recommendations'].append("Consider data cleaning and preprocessing")
        
        return validation
    
    def transform_new_data(self, df):
        """
        Transform new data using fitted preprocessors.
        
        Args:
            df: New data DataFrame
            
        Returns:
            Transformed feature matrix
        """
        
        if self.processed_features is None:
            raise ValueError("Preprocessors not fitted. Run preprocess_data first.")
        
        # Select same features used in training
        X = df[self.processed_features].copy()
        
        # Handle missing values
        if 'numeric' in self.imputers:
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                X[numeric_columns] = self.imputers['numeric'].transform(X[numeric_columns])
        
        if 'categorical' in self.imputers:
            categorical_columns = X.select_dtypes(include=['object']).columns
            if len(categorical_columns) > 0:
                X[categorical_columns] = self.imputers['categorical'].transform(X[categorical_columns])
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in X.columns:
                X[col] = encoder.transform(X[col].astype(str))
        
        # Convert to numpy and scale
        X_array = X.values.astype(float)
        X_scaled = self.scaler.transform(X_array)
        
        # Apply feature selection if used
        if self.feature_selector is not None:
            X_scaled = self.feature_selector.transform(X_scaled)
        
        return X_scaled