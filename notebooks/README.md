# Exploratory Data Analysis and Modeling Notebooks
This folder contains exploratory and early-stage modeling work that formed the foundation for the final end-to-end pipeline. These notebooks document data loading, cleaning, feature engineering, visualization, and initial machine learning experiments for the **Analysis of Delayed Flights** project completed as part of my Master's program.

## Contents
- `notebook_delayed_flights.ipynb`  
  → Primary EDA and modeling notebook using January 2019 and 2020 flight data across the United States  
- `script_delayed_flights.py`  
  → Python script version of the notebook for convenient CLI or pipeline use

## Key EDA Highlights
- **Target Label Distribution:** Delay rates by departure and arrival (separated by year: 2019 vs. 2020)
- **Temporal Patterns:**  
  - Delay frequency by day of the week  
  - Delay frequency by day of the month  
- **Airport & Carrier Impact:**  
  - Top origin and destination airports by delay percentage  
  - Carriers most frequently associated with delays  
- **Distance vs Delay:**  
  - Histogram showing that longer flights tend to have fewer delays

## Feature Engineering
- Merged datasets from 2019 and 2020
- Handled missing values and imbalanced labels
- Created custom time block features (`ARR_TIME_BLK`, `DEP_TIME_BLK`)
- Simplified airport and carrier identifiers
- Converted categorical variables with `LabelEncoder`

## Modeling Workflow
- Random undersampling to reduce dataset size
- Evaluated classifiers:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Gradient Boosting
  - SVM, SGD, Decision Trees
- Cross-validation using Accuracy, Precision, Recall and F1 Score
- Final model tuning using `GridSearchCV` on the Random Forest classifier

## Model Insights
- Confusion matrix visualizations
- Feature importance bar chart
- Evaluation metrics on both training and test datasets

## Notes
These notebooks are intended for exploratory research and complement the production-ready pipeline found in the `app/` and `airflow/` directories.

**Dataset Source:** [Flight Delay Prediction on Kaggle](https://www.kaggle.com/datasets/divyansh22/flight-delay-prediction)  
The CSV files used are available in the `data/raw/` folder for easy reference.