# Model Training Guide

## Overview
This module implements the **Adaptive Boosting (AdaBoost) Classifier with Random Forest Estimator** achieving ~92% accuracy on the back-order prediction dataset.

## Quick Start

### Train the Model
```python
from src.models.train import train_and_save_model

# Train with default settings
metadata = train_and_save_model(
    use_rfe=True,           # Use RFE for feature selection
    n_features=10,          # Select top 10 features
    test_size=0.2,          # 80/20 train/val split
    model_path='saved_models/final_model.pkl'
)
```

### Or Run from Command Line
```bash
python src/models/train.py
```

## Model Architecture

### Base Estimator: Random Forest
- **n_estimators**: 100 trees
- **max_depth**: 10
- **min_samples_split**: 5
- **min_samples_leaf**: 2

### AdaBoost Configuration
- **n_estimators**: 50 weak learners
- **learning_rate**: 1.0
- **algorithm**: SAMME (required for non-probability estimators)

## Training Process

1. **Data Loading**: Loads raw CSV files (Training_BOP.csv, Testing_BOP.csv)
2. **Preprocessing**: 
   - Handles missing values
   - Encodes categorical features
   - Performs RFE feature selection
3. **Train/Val Split**: Stratified split to maintain class distribution
4. **Model Training**: Fits AdaBoost with RF estimator
5. **Evaluation**: Computes accuracy, precision, recall, F1-score
6. **Model Saving**: Saves model and metadata as pickle files

## Performance Metrics

**Target Performance:**
- Accuracy: ~92%
- Precision: High precision for backorder prediction
- Recall: Balanced recall across classes
- F1-Score: Weighted F1 for imbalanced data

**Output Example:**
```
📊 Performance Metrics:
   Training Accuracy:   0.9450 (94.50%)
   Validation Accuracy: 0.9215 (92.15%)
   Precision:           0.9180
   Recall:              0.9215
   F1-Score:            0.9185
```

## Files Generated

### Model File
- `saved_models/final_model.pkl` - Trained model (pickle format)

### Metadata File
- `saved_models/final_model_metadata.pkl` - Contains:
  - Training date and time
  - Performance metrics
  - Feature names
  - Hyperparameters
  - Training configuration

## Loading a Trained Model

```python
from src.models.train import load_trained_model

# Load the model
model = load_trained_model('saved_models/final_model.pkl')

# Use for predictions
predictions = model.predict(X_new)
```

## Testing

### Quick Test (10K samples)
```bash
python test_train_quick.py
```

### Run Unit Tests
```bash
pytest tests/test_train.py -v
```

### Full Training (1.6M+ samples)
```bash
# Warning: Takes 10-30 minutes
python src/models/train.py
```

## Integration with Preprocessing

The training script automatically uses your preprocessing pipeline:
- `src/data/preprocess.py::load_data()` - Loads raw data
- `src/data/preprocess.py::preprocess_data()` - Cleans and transforms

## Hyperparameter Tuning

To experiment with different hyperparameters, modify the model definition in `train_and_save_model()`:

```python
# Example: More conservative Random Forest
rf_estimator = RandomForestClassifier(
    n_estimators=200,      # More trees
    max_depth=15,          # Deeper trees
    min_samples_split=10,  # More conservative splits
    random_state=42,
    n_jobs=-1
)

# Example: Different AdaBoost configuration
final_model = AdaBoostClassifier(
    estimator=rf_estimator,
    n_estimators=100,      # More boosting rounds
    learning_rate=0.5,     # Lower learning rate
    random_state=42,
    algorithm='SAMME'
)
```

## Troubleshooting

### Low Accuracy (<90%)
- Check data quality (missing values, outliers)
- Try different `n_features` in RFE
- Adjust hyperparameters
- Increase training data size

### Imbalanced Classes Warning
- Normal for this dataset (~0.7% backorder rate)
- Use stratified splitting (already implemented)
- Consider class weights if needed

### Memory Issues
- Reduce `n_estimators` in Random Forest
- Use smaller sample for initial testing
- Process in batches if needed

## Next Steps

After training:
1. Review performance metrics
2. Implement prediction pipeline (`src/models/predict.py`)
3. Deploy model for real-time predictions
4. Set up monitoring for model drift

## Contributors
- Aditya (Data Preprocessing & Model Training)
- Aryan (CI/CD & Testing Infrastructure)
