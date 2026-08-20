# Titanic Passenger Survival Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/fadlyfebro/titanic-survival-classification-pipeline)

## 1. Project Overview
An end-to-end, industry-standard machine learning pipeline designed to predict passenger survival on the RMS Titanic. By applying data science techniques and classification algorithms, this project uncovers patterns in passenger demographics and socioeconomic status to build an accurate predictive model. 

The workflow covers everything from exploratory data analysis (EDA) and robust feature engineering to Stratified 5-Fold Cross-Validation and final model serialization.

## 2. Problem Statement & Objectives
The sinking of the RMS Titanic is an infamous maritime disaster where over 1,500 people died due to insufficient lifeboats. This project aims to:
* Analyze demographic and socioeconomic factors influencing survival rates[cite: 3].
* Engineer informative features to capture relational and social dynamics[cite: 3].
* Benchmark multiple classification algorithms: Logistic Regression, Random Forest, and Gradient Boosting[cite: 3].
* Evaluate models using Accuracy, F1-Score, and ROC-AUC metrics[cite: 3].

## 3. Key Features & Methodology
* **Domain-Driven Feature Engineering:** Extraction of passenger titles (`Title`) from names, computation of family sizes (`FamilySize` = `SibSp` + `Parch` + 1), and binary solo traveler status (`IsAlone`)[cite: 3].
* **Robust Preprocessing Pipeline:** Handled missing values using group-based median imputation for `Age` (grouped by `Sex` and `Pclass`), mode imputation for `Embarked`, and median imputation for `Fare`[cite: 3].
* **Zero-Leakage Architecture:** Built a modular Scikit-Learn `ColumnTransformer` incorporating `StandardScaler` for numerical features and `OneHotEncoder` for categorical features to prevent data leakage[cite: 3].
* **Algorithm Benchmarking:** Rigorously tested models using Stratified 5-Fold Cross-Validation[cite: 3]. 
* **Deployment Ready:** The highest-performing model is serialized using `joblib` for seamless downstream inference[cite: 3].

## 4. Tech Stack
* **Language:** Python 3
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Model Serialization:** Joblib
* **Data Visualization:** Matplotlib, Seaborn

## 5. Repository Structure
```text
├── LICENSE
├── README.md
├── gender_submission.csv
├── test.csv                          # Testing dataset (418 records)
├── train.csv                         # Training dataset (891 records)
├── titanic-survival-classification-pipeline.ipynb  # Main ML Workflow Notebook
└── titanic_best_pipeline.joblib      # Serialized production pipeline
```
## 6. Key Insights & Results
Based on the Exploratory Data Analysis and Model Evaluation, several key insights were discovered:

Gender is the strongest predictor: Females had a survival rate of ~74.2%, while males only had a ~18.9% survival rate[cite: 3].

Socioeconomic Status matters: First-class passengers (Pclass 1) had priority lifeboat access and the highest survival rate, whereas third-class passengers had the lowest[cite: 3].

Family Dynamics: Small family units (2-4 members) had significantly higher survival odds compared to solo passengers and large families (>4 members)[cite: 3].

Best Model: Gradient Boosting achieved the best generalization performance, delivering an Accuracy of ~84% and an AUC-ROC score exceeding 0.88[cite: 3].

## 7. Installation & Usage
Clone the repository:
```text
git clone [https://github.com/your-username/titanic-ml-pipeline.git](https://github.com/your-username/titanic-ml-pipeline.git)
cd titanic-ml-pipeline
```
Install dependencies:
Ensure you have the required libraries installed.
```text
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```
Running Inference in Production:
To generate predictions on new data without retraining, you can load the serialized pipeline artifact:
```text
import joblib
import pandas as pd

# Load the optimized Gradient Boosting pipeline
pipeline = joblib.load('titanic_best_pipeline.joblib')

# Load new data
new_data = pd.read_csv('test.csv')

# The pipeline automatically handles feature engineering, scaling, and encoding
predictions = pipeline.predict(new_data)
print(predictions)
```

## 8. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
