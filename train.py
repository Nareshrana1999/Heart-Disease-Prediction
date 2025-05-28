import os
import warnings
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from xgboost import XGBClassifier
from data_preprocessing import load_data, preprocess_data

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Redirect stderr to suppress XGBoost warnings
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# Set XGBoost to not show warnings
os.environ['XGBOOST_VERBOSE'] = '0'

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and return the best one"""
    print("Training different models...")
    print("Models will be trained in this order (fastest to slowest):")
    print("1. Logistic Regression")
    print("2. Naive Bayes")
    print("3. Decision Tree")
    print("4. K-Nearest Neighbors")
    print("5. Random Forest")
    print("6. Gradient Boosting")
    print("7. XGBoost")
    print("8. Support Vector Machine")
    
    # All models with optimized parameters - in order of expected training time (fastest first)
    models = {
        'logistic_regression': LogisticRegression(
            max_iter=1000, 
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            random_state=42, 
            n_jobs=-1
        ),
        'naive_bayes': GaussianNB(
            var_smoothing=1e-9
        ),
        'decision_tree': DecisionTreeClassifier(
            max_depth=5, 
            min_samples_split=10, 
            min_samples_leaf=5,
            random_state=42
        ),
        'knn': KNeighborsClassifier(
            n_neighbors=5, 
            weights='distance', 
            n_jobs=-1,
            algorithm='auto'
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42, 
            n_jobs=-1, 
            verbose=0
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=200, 
            max_depth=3, 
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42, 
            verbose=0
        ),
        'xgboost': XGBClassifier(
            n_estimators=200, 
            max_depth=5, 
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42, 
            verbosity=0,
            use_label_encoder=False, 
            eval_metric='logloss',
            n_jobs=-1
        ),
        'svm': SVC(
            probability=True, 
            random_state=42, 
            kernel='rbf', 
            C=1.0, 
            gamma='scale',
            cache_size=1000
        )
    }
    
    best_model = None
    best_score = 0
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Print header
    print("\n{:<20} | {:<8} | {:<8} | {:<8} | {:<8} | {:<10}".format(
        "Model", "Accuracy", "ROC AUC", "F1", "Time (s)", "Status"
    ))
    print("-" * 70)
    
    import time
    
    for name, model in models.items():
        start_time = time.time()
        print("Training {}...".format(name), end='\r')
        with HiddenPrints():
            model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate training time
        train_time = time.time() - start_time
        
        # Print results with status
        status = "✅ Done"
        print("{:<20} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.2f} | {:<10}".format(
            name.upper(), accuracy, roc_auc, f1, train_time, status
        ))
        
        # Save the model
        model_path = f'models/{name}_model.pkl'
        joblib.dump(model, model_path)
        print(f"Saved {name} model to {model_path}")
        print("-" * 70)
        print(f"Saved {name} to {model_path}")
        
        # Track best model
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = name
    
    print(f"\nBest model: {best_model} with ROC AUC: {best_score:.4f}")
    return best_model, best_score

if __name__ == "__main__":
    print("Starting model training...")
    
    # Load data
    print("\n1. Loading data...")
    df = load_data()
    
    if df is not None:
        try:
            # Preprocess data
            print("2. Preprocessing data...")
            X_train, X_test, y_train, y_test, _ = preprocess_data(df)
            
            print(f"\nDataset Summary:")
            print(f"- Training samples: {X_train.shape[0]}")
            print(f"- Test samples: {X_test.shape[0]}")
            print(f"- Number of features: {X_train.shape[1]}")
            
            # Train models
            print("\n3. Training models...")
            print("This may take a few minutes. Please wait...\n")
            best_model, best_score = train_models(X_train, y_train, X_test, y_test)
            
            print("\n4. Training complete!")
            print(f"Best model: {best_model} (ROC AUC: {best_score:.4f})")
            
        except Exception as e:
            print("\nAn error occurred during training:")
            print(str(e))
            import traceback
            traceback.print_exc()
    else:
        print("\nError: Could not load data. Please check if the data file exists at 'data/heart_disease_cleveland.csv'")
        print("You can download it from: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data")
