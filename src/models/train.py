"""
Model Training Script
Trains the Adaptive Boosting Classifier with Random Forest Estimator
Target Accuracy: ~92% as per project report
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)

# Import preprocessing functions
from src.data.preprocess import load_data, preprocess_data


def train_and_save_model(
    use_rfe=True, 
    n_features=10,
    test_size=0.2,
    model_path='saved_models/final_model.pkl',
    save_encoder=True
):
    """
    Trains the Adaptive Boosting with Random Forest model and saves it.
    
    Args:
        use_rfe (bool): Whether to use RFE for feature selection
        n_features (int): Number of features to select if using RFE
        test_size (float): Proportion of data for testing
        model_path (str): Path to save the trained model
        save_encoder (bool): Whether to save the preprocessing encoder
    
    Returns:
        dict: Training metrics and model info
    """
    
    print("=" * 80)
    print("🚀 GROCERY AVAILABILITY - MODEL TRAINING")
    print("=" * 80)
    
    # 1. Load and Preprocess Data
    print("\n📊 Step 1: Loading Data...")
    train_df, test_df = load_data()
    
    print("\n🔧 Step 2: Preprocessing Data...")
    X_full, y_full, X_test_holdout = preprocess_data(
        train_df, 
        test_df, 
        use_rfe=use_rfe, 
        n_features=n_features
    )
    
    # 2. Split into Train/Validation
    print(f"\n✂️ Step 3: Splitting data (test_size={test_size})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, 
        test_size=test_size, 
        random_state=42,
        stratify=y_full  # Maintain class distribution
    )
    
    print(f"   Train set: {X_train.shape[0]:,} samples")
    print(f"   Validation set: {X_val.shape[0]:,} samples")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Class distribution (train): {dict(pd.Series(y_train).value_counts())}")
    
    # 3. Define the Model Architecture
    print("\n🤖 Step 4: Defining Model Architecture...")
    print("   Model: AdaBoost with Random Forest Estimator")
    
    # Random Forest as base estimator
    rf_estimator = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # AdaBoost with RF estimator
    final_model = AdaBoostClassifier(
        estimator=rf_estimator,
        n_estimators=50,
        learning_rate=1.0,
        random_state=42,
        algorithm='SAMME'  # Required for non-probability estimators
    )
    
    print(f"   Base Estimator: Random Forest (n_estimators=100)")
    print(f"   AdaBoost: n_estimators=50, learning_rate=1.0")
    
    # 4. Train the Model
    print("\n🎓 Step 5: Training Model...")
    print("   This may take several minutes on the full dataset...")
    
    start_time = datetime.now()
    final_model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"   ✅ Training completed in {training_time:.2f} seconds")
    
    # 5. Evaluate on Validation Set
    print("\n📈 Step 6: Evaluating Model Performance...")
    
    # Training set metrics
    y_train_pred = final_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Validation set metrics
    y_val_pred = final_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
    
    print("\n📊 Performance Metrics:")
    print(f"   Training Accuracy:   {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"   Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"   Precision:           {val_precision:.4f}")
    print(f"   Recall:              {val_recall:.4f}")
    print(f"   F1-Score:            {val_f1:.4f}")
    
    if val_accuracy >= 0.90:
        print(f"   🎉 Target accuracy achieved! (≥90%)")
    else:
        print(f"   ⚠️ Accuracy below target (expected ~92%)")
    
    # Detailed classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=['No Backorder', 'Backorder']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_val_pred)
    print("\n🔲 Confusion Matrix:")
    print(f"   True Negatives:  {cm[0][0]:,}")
    print(f"   False Positives: {cm[0][1]:,}")
    print(f"   False Negatives: {cm[1][0]:,}")
    print(f"   True Positives:  {cm[1][1]:,}")
    
    # 6. Save the Model
    print(f"\n💾 Step 7: Saving Model...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    with open(model_path, 'wb') as file:
        pickle.dump(final_model, file)
    print(f"   ✅ Model saved to: {model_path}")
    
    # Save model metadata
    metadata = {
        'model_type': 'AdaBoost with Random Forest',
        'trained_date': datetime.now().isoformat(),
        'training_time_seconds': training_time,
        'train_accuracy': train_accuracy,
        'validation_accuracy': val_accuracy,
        'precision': val_precision,
        'recall': val_recall,
        'f1_score': val_f1,
        'n_features': X_train.shape[1],
        'feature_names': list(X_train.columns),
        'n_training_samples': X_train.shape[0],
        'n_validation_samples': X_val.shape[0],
        'use_rfe': use_rfe,
        'n_features_selected': n_features,
        'hyperparameters': {
            'rf_n_estimators': 100,
            'adaboost_n_estimators': 50,
            'learning_rate': 1.0,
            'random_state': 42
        }
    }
    
    metadata_path = model_path.replace('.pkl', '_metadata.pkl')
    with open(metadata_path, 'wb') as file:
        pickle.dump(metadata, file)
    print(f"   ✅ Metadata saved to: {metadata_path}")
    
    print("\n" + "=" * 80)
    print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return metadata


def load_trained_model(model_path='saved_models/final_model.pkl'):
    """
    Load a previously trained model.
    
    Args:
        model_path (str): Path to the saved model
    
    Returns:
        model: Trained model object
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # Load metadata if available
    metadata_path = model_path.replace('.pkl', '_metadata.pkl')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as file:
            metadata = pickle.load(file)
        print(f"📦 Model loaded: {metadata['model_type']}")
        print(f"   Trained: {metadata['trained_date']}")
        print(f"   Validation Accuracy: {metadata['validation_accuracy']:.4f}")
    
    return model


if __name__ == '__main__':
    """
    Main execution: Train and save the final model
    """
    print("\n🎯 Training Adaptive Boosting with Random Forest Estimator")
    print("   Target: ~92% Accuracy\n")
    
    # Train the model with default settings
    metadata = train_and_save_model(
        use_rfe=True,
        n_features=10,
        test_size=0.2,
        model_path='saved_models/final_model.pkl'
    )
    
    print(f"\n✨ Model is ready for predictions!")
    print(f"   Use: from src.models.predict import predict")
