# Transaction Fraud Detection System

A production-level machine learning system for detecting fraudulent credit card transactions using anomaly detection and classification algorithms.

## 🎯 Project Overview

This project implements a comprehensive fraud detection system that:
- Handles highly imbalanced transaction data
- Applies multiple ML algorithms (Logistic Regression, Random Forest, XGBoost, Isolation Forest)
- Uses SMOTE for handling class imbalance
- Provides real-time prediction capabilities
- Includes extensive model evaluation and monitoring

## 📊 Dataset

**Source:** [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains credit card transactions made by European cardholders in September 2013. It presents transactions that occurred over two days, with 492 frauds out of 284,807 transactions (highly imbalanced - 0.172% fraud rate).

## 🏗️ Project Structure

```
Transaction-Fraud-Detection-System/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Processed and feature-engineered data
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── data/                   # Data loading and preprocessing
│   ├── features/               # Feature engineering
│   ├── models/                 # Model training and evaluation
│   └── utils/                  # Helper functions and utilities
├── tests/                      # Unit tests
├── configs/                    # Configuration files
├── logs/                       # Training and application logs
├── models/                     # Saved model artifacts
├── requirements.txt            # Python dependencies
└── README.md
```

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)
- Kaggle account (for dataset download)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/levdalba/Transaction-Fraud-Detection-System.git
cd Transaction-Fraud-Detection-System
```

2. **Create and activate virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Setup Kaggle API credentials**
```bash
# Get your kaggle.json from https://www.kaggle.com/settings/account
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

5. **Download dataset**
```bash
python src/data/download_dataset.py
```

## 💻 Usage

### Data Exploration
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

### Data Processing
Process raw data with validation, cleaning, and splitting:
```bash
python -m src.data.process_data --input data/raw/creditcard.csv --output data/processed
```

Output:
- `data/processed/train.csv` - Training set (70%)
- `data/processed/val.csv` - Validation set (15%)
- `data/processed/test.csv` - Test set (15%)
- `data/processed/scalers/` - Fitted scalers for Amount and Time
- `data/processed/metadata.yaml` - Processing metadata

### Model Training
```bash
python src/models/train.py --config configs/model_config.yaml
```

### Making Predictions
```bash
python src/models/predict.py --input data/test_transactions.csv
```

## 🧪 Running Tests
```bash
pytest tests/ -v --cov=src
```

## 📈 Model Performance

Performance metrics and model comparisons will be documented after training.

## 🔧 Technologies Used

- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, XGBoost, imbalanced-learn
- **Visualization:** matplotlib, seaborn, plotly
- **Experiment Tracking:** MLflow (planned)
- **Testing:** pytest, pytest-cov
- **Logging:** Python logging, colorlog
- **Configuration:** PyYAML
- **Model Persistence:** joblib

## ✨ Key Features

### Production-Level Data Pipeline
- **DataLoader**: Robust CSV loading with validation and error handling
- **DataValidator**: Comprehensive quality checks (schema, types, ranges, duplicates, class balance)
- **DataPreprocessor**: 
  - Configurable duplicate removal
  - Missing value handling strategies
  - StandardScaler for Amount and Time features
  - Stratified train/val/test splitting (70/15/15)
  - Scaler persistence for inference
- **DataPipeline**: End-to-end orchestration with CLI interface

### Data Quality Assurance
- 10+ validation checks ensuring data integrity
- Automated detection of statistical anomalies
- Configurable thresholds for quality metrics
- Comprehensive logging at every step

### Testing & Documentation
- Unit tests with pytest (100% pass rate)
- Fixtures for reproducible test data
- Clear docstrings and type hints
- Professional git workflow (feature branches, atomic commits)

## 🌿 Git Workflow

This project follows a structured branching strategy:
- `main` - Production-ready code
- `dev` - Integration and testing
- `feature/*` - Individual feature development

## 📝 Development Roadmap

- [x] Project setup and structure
- [x] Environment configuration
- [x] Exploratory Data Analysis
- [x] Data preprocessing pipeline (with validation and testing)
- [ ] Feature engineering module
- [ ] Baseline model development (Logistic Regression, Random Forest)
- [ ] Advanced model experimentation (XGBoost, Isolation Forest, Neural Nets)
- [ ] Model evaluation and comparison (with cross-validation)
- [ ] Hyperparameter tuning (Optuna/GridSearch)
- [ ] Model ensemble and stacking
- [ ] MLflow experiment tracking integration
- [ ] Real-time prediction API (FastAPI)
- [ ] Model monitoring and drift detection
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Dockerization
- [ ] Deployment guide and documentation

## 👤 Author

**Levan Dalba**
- GitHub: [@levdalba](https://github.com/levdalba)

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset provided by the Machine Learning Group - ULB
- Inspired by real-world fraud detection systems in fintech

---

**Note:** This is an educational project demonstrating production-level ML system design for portfolio purposes.
