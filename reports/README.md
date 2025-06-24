# Project Report and Presentation
This folder contains the formal deliverables submitted for academic evaluation as part of my Master’s program in Data Science. These documents summarize the full analytical process — from data exploration to model development and evaluation of the **Analysis of Delayed Flights** project.  

## Contents
- `report_delayed_flights.pdf`  
  → Comprehensive technical report covering methodology, modeling, and insights
- `presentation_delayed_flights.pdf`  
  → Slide deck used to present project outcomes to faculty and peers

## Report Overview
### **Objective**
To build a predictive model for flight arrival delays at U.S. destination airports in January, using data from 2019 and 2020 and leveraging classification-based machine learning models.

### **Research Question**
Can we predict arrival delays using variables such as day of the week, carrier, origin, destination, and flight distance?

## Methodology Summary
### Data Preprocessing
- Merged January 2019 and 2020 flight datasets from the U.S. Bureau of Transportation Statistics
- Addressed missing and misleading values (e.g., diverted and canceled flights)
- Created time block features (`ARR_TIME_BLK`, `DEP_TIME_BLK`)
- Dropped correlated or redundant columns
- Encoded categorical variables with `LabelEncoder`

### Exploratory Analysis
- Temporal delay patterns (by weekday and day of month)
- Delay frequency by airport and airline
- Minimal correlation between distance and delays
- Heatmaps revealed inter-feature correlations

### Modeling Approach
- Used a 10% sample of the full dataset for faster training
- Split data into 70:30 train-test sets; applied oversampling for class balance
- Trained and compared 7 classification algorithms:
  - Logistic Regression, SGD, Decision Tree, Random Forest, SVM, Gradient Boosting, and XGBoost
- Evaluated models using 5-fold cross-validation

### Model Selection
- Random Forest outperformed other models
- Tuned using `GridSearchCV` for hyperparameter optimization
- Final model achieved 1.00 training accuracy and ~0.92 test accuracy

## Conclusions
- **Best-performing model:** Random Forest Classifier
- **Top feature:** Departure delay status was the strongest predictor for arrival delay
- **Impact:** Useful for improving planning, reducing disruptions, and enhancing customer satisfaction

## Recommendations for Future Work
- Expand analysis to year-round data to capture seasonal variation
- Integrate weather-related features (e.g., wind speed, snow, visibility)
- Explore real-time applications with streaming data and airline APIs

## References
- Los Angeles Times (2022) – Trends in airline delays  
- [Kaggle – Flight Delay Prediction Dataset](https://www.kaggle.com/datasets/divyansh22/flight-delay-prediction)

## Related Project Areas
- Full model pipeline: see `notebooks/` and `app/` folders  
- Production workflow: see `airflow/` for DAG-based orchestration  

These deliverables serve as a formal summary of the exploratory and production-level work behind the flight delay prediction system.