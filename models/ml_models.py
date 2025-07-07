import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
import joblib

class PTSDMLModels:
    """
    Machine Learning models for PTSD prediction based on research findings.
    Implements SVM, ANN, Decision Trees, Gaussian Naive Bayes, and ensemble methods.
    """
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.scalers = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        
        # Initialize models based on research findings
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models with optimal hyperparameters from research."""
        
        # SVM - Best performer in neuroimaging studies (82-90% accuracy)
        self.models['SVM'] = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ))
        ])
        
        # Artificial Neural Network - Best for PCL-5 data (90% accuracy)
        self.models['ANN'] = Pipeline([
            ('scaler', StandardScaler()),
            ('ann', MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=1000,
                random_state=42
            ))
        ])
        
        # Decision Tree - Interpretable model
        self.models['Decision_Tree'] = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Gaussian Naive Bayes - Best precision in some studies
        self.models['Gaussian_NB'] = Pipeline([
            ('scaler', StandardScaler()),
            ('gnb', GaussianNB())
        ])
        
        # Random Forest - Strong ensemble method
        self.models['Random_Forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Gradient Boosting - Advanced ensemble
        self.models['Gradient_Boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
    
    def train_models(self, X, y, test_size=0.2, cv_folds=5):
        """
        Train all models and evaluate performance.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
        
        Returns:
            Dictionary with model results
        """
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = None
            
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except:
                # Some models might not support predict_proba
                pass
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Calculate AUC if probabilities available
            auc = None
            if y_pred_proba is not None:
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    pass
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            
            # Store results
            results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'predictions': y_pred,
                'actual': y_test,
                'probabilities': y_pred_proba
            }
            
            # Update best model
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model_name
        
        self.results = results
        self.trained_models = {name: result['model'] for name, result in results.items()}
        
        return results
    
    def hyperparameter_tuning(self, X, y, model_name='SVM'):
        """
        Perform hyperparameter tuning for specified model.
        
        Args:
            X: Feature matrix
            y: Target labels
            model_name: Name of model to tune
        
        Returns:
            Best parameters and score
        """
        
        if model_name == 'SVM':
            param_grid = {
                'svm__C': [0.1, 1, 10, 100],
                'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'svm__kernel': ['rbf', 'linear', 'poly']
            }
            model = self.models['SVM']
            
        elif model_name == 'ANN':
            param_grid = {
                'ann__hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
                'ann__alpha': [0.0001, 0.001, 0.01],
                'ann__learning_rate': ['constant', 'adaptive']
            }
            model = self.models['ANN']
            
        elif model_name == 'Random_Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10]
            }
            model = self.models['Random_Forest']
            
        else:
            raise ValueError(f"Hyperparameter tuning not implemented for {model_name}")
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
    
    def predict(self, X, model_name=None):
        """
        Make predictions using trained model.
        
        Args:
            X: Feature matrix
            model_name: Specific model to use (if None, uses best model)
        
        Returns:
            Predictions and probabilities
        """
        
        if model_name is None:
            model_name = self.best_model
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        predictions = model.predict(X)
        
        try:
            probabilities = model.predict_proba(X)[:, 1]
        except:
            probabilities = None
        
        return predictions, probabilities
    
    def get_feature_importance(self, model_name=None):
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Name of model
        
        Returns:
            Feature importance array
        """
        
        if model_name is None:
            model_name = self.best_model
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        # Extract the actual classifier from pipeline if needed
        if hasattr(model, 'named_steps'):
            # It's a pipeline
            classifier = None
            for step_name, step in model.named_steps.items():
                if hasattr(step, 'feature_importances_'):
                    classifier = step
                    break
        else:
            classifier = model
        
        if hasattr(classifier, 'feature_importances_'):
            return classifier.feature_importances_
        else:
            return None
    
    def save_models(self, filepath):
        """Save trained models to file."""
        model_data = {
            'trained_models': self.trained_models,
            'results': self.results,
            'best_model': self.best_model,
            'best_score': self.best_score
        }
        joblib.dump(model_data, filepath)
    
    def load_models(self, filepath):
        """Load trained models from file."""
        model_data = joblib.load(filepath)
        self.trained_models = model_data['trained_models']
        self.results = model_data['results']
        self.best_model = model_data['best_model']
        self.best_score = model_data['best_score']