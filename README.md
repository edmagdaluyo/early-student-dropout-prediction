## Explainable Machine Learning for Early Dropout Prediction and Student Segmentation in Higher Education

## Overview

This repository contains the Google Colab implementation of an AI/ML capstone project
that develops an explainable machine learning framework for early student-dropout
prediction and student segmentation in higher education.

The study uses the UCI **Predict Students' Dropout and Academic Success** dataset.
It integrates supervised classification, feature engineering, model calibration,
classification-threshold optimization, explainable artificial intelligence, fairness
assessment, and unsupervised clustering.

## Study Objectives

1. Develop models for predicting student dropout at different intervention horizons.
2. Compare Logistic Regression, Decision Tree, Random Forest, Gradient Boosting,
   and Support Vector Machine classifiers.
3. Identify influential predictors through feature importance, SHAP, and partial
   dependence analysis.
4. Evaluate model performance across selected demographic and socioeconomic groups.
5. Segment students using K-Means, DBSCAN, and Agglomerative Clustering.
6. Translate model outputs into operational indicators for student-support planning.

## Dataset

- **Source:** UCI Machine Learning Repository
- **Dataset:** Predict Students' Dropout and Academic Success
- **Observations:** 4,424 students
- **Original variables:** 37
- **Outcome classes:** Graduate, Dropout, and Enrolled
- **Modeling target:** Binary dropout outcome
- **Prediction horizons:** Admission, after the first semester, and after the second semester

The notebook downloads the dataset automatically to:

```text
data/raw/student_dropout_data.csv
```

## Methods

The analysis includes:

- Data-quality assessment and exploratory data analysis
- Feature engineering and prediction-horizon construction
- Preprocessing with imputation, scaling, and one-hot encoding
- Mutual-information feature selection
- Principal Component Analysis
- Stratified train-test splitting and cross-validation
- Grid-search hyperparameter optimization
- Probability calibration and threshold optimization
- Classification performance evaluation
- SHAP, permutation importance, and partial dependence analysis
- Group-based fairness assessment
- K-Means, DBSCAN, and Agglomerative Clustering
- Operational KPI calculation

## Main Operational Model

The operational model uses information available after the first semester. Based on
the completed study run, Logistic Regression provided a strong balance of predictive
performance, probability quality, interpretability, and practical usefulness.

Reported held-out test performance at the selected operating threshold:

| Metric | Value |
|---|---:|
| Accuracy | 84.3% |
| Balanced Accuracy | 84.5% |
| Precision | 71.4% |
| Recall | 85.2% |
| F1-score | 77.7% |
| ROC-AUC | 0.913 |
| PR-AUC | 0.877 |
| Brier score | 0.098 |

These values should be interpreted as results from this dataset and validation design,
not as evidence of causal effects or guaranteed performance at another institution.

## Repository Structure

```text
.
├── Eduardo_Magdaluyo_Jr_Capstone_Project_GitHub_Colab.ipynb
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
│   ├── figures/
│   └── tables/
├── configs/
├── notebooks/
└── slides/
```

## Running the Notebook in Google Colab

1. Open the `.ipynb` file in Google Colab.
2. Select **Runtime > Run all**.
3. Allow the package-installation cell to finish.
4. The dataset will be downloaded automatically when it is not yet available.
5. Generated figures, tables, models, and configuration files will be saved in their
   corresponding project folders.
6. Download the generated outputs from the Colab Files panel before ending the runtime,
   because files stored under `/content` are temporary.

## Reproducibility

The notebook uses a fixed random seed, a stratified 80/20 train-test split,
stratified five-fold cross-validation, and saved run configuration metadata.

## Author

**Eduardo R. Magdaluyo, Jr.**  
Capstone Project  
Post Graduate Diploma in AI/ML  
Asian Institute of Management

## License

This project is distributed under the MIT License. See `LICENSE` for details.
"""

readme_path = ROOT / "README.md"
readme_path.write_text(readme_content, encoding="utf-8")

print(f"README.md created at:\n{readme_path.resolve()}")
