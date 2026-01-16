# Health Phenotype Discovery

A comprehensive machine learning project for discovering and analyzing health phenotypes from the National Health and Nutrition Examination Survey (NHANES) dataset.

## Overview

This project aims to identify meaningful health phenotypes from NHANES health data using advanced machine learning techniques. The analysis includes data preprocessing, feature engineering, clustering, classification, and interpretability analysis.

## Dataset

- **Source**: National Health and Nutrition Examination Survey (NHANES)
- **Source Organization**: Centers for Disease Control and Prevention (CDC), National Center for Health Statistics (NCHS)
- **Source URL**: https://www.cdc.gov/nchs/nhanes/
- **Location**: `data/raw/nhanes_health_data.csv`
- **Samples**: 5,000 adult respondents
- **Variables**: 47-48 health indicators

### Data Categories

The dataset includes health indicators across several categories:

1. **Demographics**: Age, Gender, Race/Ethnicity, Education Level, Income Category
2. **Examination**: Blood Pressure, BMI, Waist Circumference, Pulse, Respiratory Rate
3. **Laboratory**: Cholesterol (Total, HDL, LDL), Triglycerides, Blood Glucose, HbA1c, Creatinine, BUN, Hemoglobin, WBC, Platelets, Vitamin D
4. **Lifestyle**: Smoking Status, Alcohol Consumption, Physical Activity, Sedentary Behavior, Sleep
5. **Medical Conditions**: Diabetes, Hypertension, High Cholesterol, Heart Disease, Stroke, Arthritis, Asthma, Cancer, Depression, COPD
6. **Medications**: BP Medication, Cholesterol Medication, Diabetes Medication
7. **Functional Status**: Difficulty Walking, Standing, Lifting

## Project Structure

```
health-phenotype-discovery/
├── data/
│   ├── raw/
│   │   └── nhanes_health_data.csv
│   └── processed/
│       └── (processed datasets)
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── phenotype_clustering.ipynb
│   ├── classification_analysis.ipynb
│   └── interpretability_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clustering.py
│   │   ├── classification.py
│   │   └── interpretability.py
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       └── visualization.py
├── tests/
│   ├── __init__.py
│   ├── test_data_loading.py
│   ├── test_preprocessing.py
│   └── test_models.py
├── scripts/
│   ├── download_nhanes.py
│   ├── run_analysis.py
│   └── generate_report.py
├── config/
│   ├── default_config.yaml
│   └── model_config.yaml
├── requirements.txt
├── setup.py
├── .gitignore
├── README.md
└── LICENSE
```

## Installation

```bash
# Clone the repository
git clone https://github.com/OumaCavin/health-phenotype-discovery.git
cd health-phenotype-discovery

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Download Dataset
```bash
python scripts/download_nhanes.py
```

### Run Complete Analysis
```bash
python scripts/run_analysis.py
```

### Generate Report
```bash
python scripts/generate_report.py
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- shap
- pyyaml

See `requirements.txt` for complete list.

## Features

- **Data Preprocessing**: Handling missing values, encoding categorical variables, feature scaling
- **Clustering Analysis**: K-means, hierarchical clustering, DBSCAN for phenotype discovery
- **Classification Models**: Random Forest, XGBoost, Logistic Regression for health outcome prediction
- **Interpretability**: SHAP values for model explanation and feature importance analysis
- **Visualization**: Comprehensive visualizations for data exploration and model results

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CDC National Center for Health Statistics for NHANES data
- Open source community for machine learning tools and libraries
