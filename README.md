# 🛒 Grocery Availability Checker & Stock Predictor

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> An intelligent machine learning system for predicting grocery product backorders using Adaptive Boosting with Random Forest, achieving ~92% accuracy on 1.6M+ data points.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Contributors](#contributors)
- [License](#license)

## 🎯 Overview

This project implements a production-ready machine learning pipeline for predicting whether grocery products will go on backorder. By analyzing historical inventory, sales, and supply chain data, the system helps retailers:

- **Prevent Stockouts**: Predict which items are likely to go on backorder
- **Optimize Inventory**: Make data-driven restocking decisions
- **Reduce Costs**: Minimize overstocking and emergency shipping
- **Improve Customer Satisfaction**: Ensure product availability

### Key Achievements
- ✅ **92% Accuracy** on validation set
- ✅ **1.6M+ Training Samples** processed efficiently
- ✅ **10 Key Features** selected via RFE from 23 original features
- ✅ **Production-Ready** with full CI/CD pipeline
- ✅ **Cross-Platform** support (Windows, macOS, Linux)

## ✨ Features

### Data Processing
- **Automated Data Loading**: Handles large CSV files (100MB+)
- **Smart Preprocessing**: Missing value handling, encoding, feature engineering
- **Feature Selection**: RFE (Recursive Feature Elimination) for optimal features
- **Imbalanced Data Handling**: Stratified splitting for rare backorder events (~0.7%)

### Machine Learning
- **Model**: Adaptive Boosting (AdaBoost) with Random Forest estimator
- **Architecture**: 100 RF trees + 50 AdaBoost estimators
- **Evaluation**: Comprehensive metrics (accuracy, precision, recall, F1)
- **Persistence**: Model and metadata saving for deployment

### Development Tools
- **Automated Testing**: 15+ unit tests with pytest
- **CI/CD Pipeline**: GitHub Actions for continuous integration
- **Docker Support**: Containerized deployment ready
- **Code Quality**: Linting, formatting, and type checking

## 📁 Project Structure

```
Grocry-Availability-Checker-And-Stock-Predictor/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml              # Continuous Integration
│   │   ├── cd.yml              # Continuous Deployment
│   │   └── dependency-updates.yml
│   ├── CICD.md                 # CI/CD documentation
│   └── pull_request_template.md
├── data/
│   ├── Training_BOP.csv        # Training data (122MB, 1.69M rows)
│   └── Testing_BOP.csv         # Test data (17.5MB, 242K rows)
├── docs/
│   └── MODEL_TRAINING.md       # Training guide
├── notebooks/
│   └── J6_Experimentation.ipynb # EDA and experiments
├── saved_models/
│   ├── final_model.pkl         # Trained model
│   └── final_model_metadata.pkl # Model metadata
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── load_data.py        # Data loading utilities
│   │   └── preprocess.py       # Preprocessing pipeline
│   ├── models/
│   │   ├── train.py            # Model training
│   │   └── predict.py          # Prediction interface
│   └── main.py                 # Main application
├── tests/
│   ├── test_load_data.py       # Data loading tests
│   ├── test_preprocess.py      # Preprocessing tests
│   ├── test_train.py           # Training tests
│   └── conftest.py             # Test configuration
├── .gitignore
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker Compose setup
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── setup.sh                     # Setup script
└── README.md                    # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.11 or higher
- Git
- 8GB+ RAM recommended (for full dataset)
- 200MB+ disk space

### Setup Instructions

#### 1. Clone the Repository
```bash
git clone https://github.com/aryanprahraj/Grocry-Availability-Checker-And-Stock-Predictor.git
cd Grocry-Availability-Checker-And-Stock-Predictor
```

#### 2. Create Virtual Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Download Dataset
Download the Back Order Prediction dataset from Kaggle and place the files in the `data/` folder:
- `Training_BOP.csv` (~122MB)
- `Testing_BOP.csv` (~17.5MB)

**Dataset Source**: [Kaggle - Back Order Prediction](https://www.kaggle.com/datasets/yourusername/back-order-prediction)

## 🏃 Quick Start

### Test the Installation
```bash
# Quick test with sample data (30 seconds)
python test_preprocess_quick.py
```

### Train the Model
```bash
# Quick training test with 10K samples (1-2 minutes)
python test_train_quick.py

# Full model training with entire dataset (10-30 minutes)
python src/models/train.py
```

### Run Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_train.py -v
```

## 📊 Dataset

### Source
Back Order Prediction Dataset from Kaggle (~8K products, 1.9M total records)

### Features (23 columns)
- **Inventory Metrics**: `national_inv`, `lead_time`, `in_transit_qty`
- **Sales Data**: `sales_1_month`, `sales_3_month`, `sales_6_month`, `sales_9_month`
- **Forecasts**: `forecast_3_month`, `forecast_6_month`, `forecast_9_month`
- **Performance**: `perf_6_month_avg`, `perf_12_month_avg`
- **Risk Indicators**: `potential_issue`, `deck_risk`, `oe_constraint`, `ppap_risk`
- **Product Info**: `sku`, `stop_auto_buy`, `rev_stop`
- **Target**: `went_on_backorder` (Yes/No)

### Statistics
- **Training Set**: 1,687,861 rows × 23 columns
- **Testing Set**: 242,076 rows × 23 columns
- **Class Distribution**: ~0.7% backorder rate (highly imbalanced)
- **Missing Values**: ~115K rows removed during preprocessing

## 🤖 Model Architecture

### Pipeline
```
Raw Data → Preprocessing → Feature Selection → Model Training → Evaluation → Deployment
```

### Components

#### 1. Preprocessing (`src/data/preprocess.py`)
- **Missing Value Handling**: Remove rows with nulls
- **Categorical Encoding**: OrdinalEncoder for 7 categorical features
- **Target Encoding**: Yes/No → 1/0
- **Feature Selection**: RFE to select top 10 features

#### 2. Model (`src/models/train.py`)
- **Base Estimator**: Random Forest
  - 100 trees
  - Max depth: 10
  - Min samples split: 5
- **Boosting**: AdaBoost
  - 50 estimators
  - Learning rate: 1.0
  - Algorithm: SAMME

#### 3. Selected Features (Top 10)
1. `sku` - Product identifier
2. `national_inv` - Current inventory
3. `forecast_3_month`, `forecast_6_month`, `forecast_9_month`
4. `sales_3_month`, `sales_6_month`, `sales_9_month`
5. `perf_6_month_avg`, `perf_12_month_avg`

### Performance Metrics
```
Validation Accuracy:  92.15%
Precision:            91.80%
Recall:               92.15%
F1-Score:             91.85%
```

## 💻 Usage

### Training a New Model

```python
from src.models.train import train_and_save_model

# Train with custom parameters
metadata = train_and_save_model(
    use_rfe=True,           # Use RFE for feature selection
    n_features=10,          # Number of features to select
    test_size=0.2,          # Validation split
    model_path='saved_models/my_model.pkl'
)

# View training results
print(f"Validation Accuracy: {metadata['validation_accuracy']:.4f}")
print(f"Features: {metadata['feature_names']}")
```

### Loading and Using a Trained Model

```python
from src.models.train import load_trained_model
import pandas as pd

# Load the model
model = load_trained_model('saved_models/final_model.pkl')

# Prepare your data (same preprocessing as training)
from src.data.preprocess import load_data, preprocess_data

train_df, test_df = load_data()
X_train, y_train, X_test = preprocess_data(train_df, test_df)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)  # If available

# Interpret results
backorder_prob = probabilities[:, 1]  # Probability of backorder
print(f"Products at risk: {(predictions == 1).sum()}")
```

### Preprocessing Only

```python
from src.data.preprocess import load_data, preprocess_data

# Load raw data
train_df, test_df = load_data()

# Preprocess
X_train, y_train, X_test = preprocess_data(
    train_df, 
    test_df, 
    use_rfe=True, 
    n_features=10
)

print(f"Training shape: {X_train.shape}")
print(f"Selected features: {list(X_train.columns)}")
```

## 🧪 Testing

### Test Structure
```
tests/
├── test_basic.py           # Basic environment tests
├── test_load_data.py       # Data loading tests
├── test_preprocess.py      # Preprocessing tests
└── test_train.py           # Training tests
```

### Running Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_train.py::test_model_training_module_exists -v

# Run tests in parallel
pytest tests/ -n auto
```

### Continuous Integration
Tests automatically run on every push and pull request via GitHub Actions.

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

#### CI Pipeline (`.github/workflows/ci.yml`)
- **Triggers**: Push, Pull Request
- **Runs on**: Ubuntu, Windows, macOS
- **Steps**:
  1. Checkout code
  2. Set up Python 3.11
  3. Install dependencies
  4. Run linting (flake8)
  5. Run tests (pytest)
  6. Generate coverage report

#### CD Pipeline (`.github/workflows/cd.yml`)
- **Triggers**: Push to main branch
- **Steps**:
  1. Build Docker image
  2. Run integration tests
  3. Deploy to staging/production

### Local Docker Development

```bash
# Build image
docker-compose build

# Run container
docker-compose up

# Run tests in container
docker-compose run app pytest tests/
```

## 👥 Contributors

### Development Team

**Aditya** 
- Data preprocessing pipeline
- Model training implementation
- Feature engineering

**Aryan** ([@aryanprahraj](https://github.com/aryanprahraj))
- CI/CD infrastructure
- Testing framework
- Docker configuration
- Project architecture

### Acknowledgments
- Dataset: Kaggle Back Order Prediction Dataset
- Framework: scikit-learn
- Inspiration: Supply chain optimization research

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Additional Documentation

- [Model Training Guide](docs/MODEL_TRAINING.md) - Detailed training documentation
- [CI/CD Implementation](CICD_IMPLEMENTATION.md) - CI/CD setup guide
- [Contributing Guidelines](.github/pull_request_template.md) - How to contribute

## 🐛 Troubleshooting

### Common Issues

**Issue: ModuleNotFoundError**
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Issue: Data files not found**
```bash
# Solution: Download dataset and place in data/ folder
# Ensure files are named: Training_BOP.csv, Testing_BOP.csv
```

**Issue: Memory error during training**
```bash
# Solution: Use sample data for testing
python test_train_quick.py

# Or reduce dataset size in preprocessing
```

**Issue: Tests fail in CI**
```bash
# Tests skip data-dependent tests automatically in CI
# Run locally with: pytest tests/ -v
```

## 🔮 Future Enhancements

- [ ] Real-time prediction API (Flask/FastAPI)
- [ ] Web dashboard for visualization
- [ ] Hyperparameter tuning with Optuna
- [ ] Feature importance visualization
- [ ] A/B testing framework
- [ ] Model monitoring and drift detection
- [ ] Multi-model ensemble approach
- [ ] Automated retraining pipeline

## 📧 Contact

For questions or suggestions:
- Open an issue on GitHub
- Contact: [@aryanprahraj](https://github.com/aryanprahraj)

## ⭐ Show Your Support

If this project helped you, please consider giving it a ⭐️!

---

**Built with ❤️ for better supply chain management**