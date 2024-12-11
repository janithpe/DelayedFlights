# Delayed Flights
 Study of Possible Factors Caused Flight Delays at Destination Airports in USA

 ## Table of Contents
 * [Introduction](#introduction)
    - [Research Question](#research-question)
    - [Dataset Description](#dataset-description)
 * [Techologies](#technologies)
 * [Methods Used](#methods-used)
 * [Project Contents](#project-contents)

 ## Introduction
 This project is a part of my Master's program in Data Science, where I embarked on an individual assignment to predict possible flight delays at destination airports specifically for the month of January. Leveraging exploratory data analysis and various machine learning classification models, the goal was to build a robust prediction model that could foresee delays based on several influencing factors.

 ### Research Question
 Using exploratory analysis and popular machine learning classification models, can we predict possible flight delays at destination airports specifically for the month of January in upcoming years, based on factors such as the day of the week, date of the month, carrier, origin, destination, and flight distance?

 ### Dataset Description
 The dataset was downloaded from [Kaggle](https://www.kaggle.com/datasets/divyansh22/flight-delay-prediction) and contains one binary target variable and 21 feature columns, including origin airport, destination airport, carrier, and flight information. It consists of nearly 1.2 million instances, originally collected from the Bureau of Transportation Statistics, Government of the United States of America. The dataset includes all flights throughout the United States in January 2019 and January 2020.

 ## Technologies
 This project was developed using Python 3.8.5 in a Jupyter Notebook environment. The following libraries and modules were utilized:
 * Pandas 1.4.1 - For data manipulation and analysis
 * Numpy 1.22.1 - For numerical computations
 * Seaborn 0.11.2 - For data visualization
 * Matplotlib 3.5.1 - For creating static, animated, and interactive visualizations
 * Sklearn 1.1.1 - For machine learning algorithms and tools
 * Xgboost 1.6.0 - For gradient boosting algorithms
 * Imblearn 0.9.1 - For handling imbalanced datasets

 ## Methods Used
 The following methods and models were employed in this project:
 * Label Encoder and Label Binarizer
 * Random Under/Over Sampling
 * Logistic Regression Classifier
 * Stochastic Gradient Descent (SGD) Classifier
 * Decision Tree Classifier
 * Random Forest Classifier
 * Support Vector Machine Classifier (SVC)
 * Gradient Boosting Classifier
 * XG Boost Classifier
 * Grid search Cross validation

 ## Project Contents
 1. Data Files Folder - Contains all the relevant datasets used in this project.
 2. Python Notebook - The Jupyter Notebook with all the code and analysis.
 3. Written Report - A concise paper summarizing the project's findings and methodologies.
 4. Presentation - A slide deck presenting the key points and results of the project.

 #### For additional details, conclusions, and recommendations of the project, please refer to the comprehensive PDF version of the [paper](https://github.com/janithpe/DelayedFlights/blob/main/Report_Analysis%20of%20Delayed%20Flights.pdf) and the [accompanying presentation](https://github.com/janithpe/DelayedFlights/blob/main/Presentation_Analysis%20of%20Delayed%20Flights.pdf).