import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

flights_jan_2019 = pd.read_csv("https://raw.githubusercontent.com/emmanueliarussi/DataScienceCapstone/master/3_MidtermProjects/ProjectFD/data/jan_2019_ontime.csv")
flights_jan_2020 = pd.read_csv("https://raw.githubusercontent.com/emmanueliarussi/DataScienceCapstone/master/3_MidtermProjects/ProjectFD/data/jan_2020_ontime.csv")

print(flights_jan_2019.shape)
flights_jan_2019.head()

print(flights_jan_2020.shape)
flights_jan_2020.head()

flights_jan_2019['YEAR'] = 2019
flights_jan_2020['YEAR'] = 2020

print(set(flights_jan_2019.columns) == set(flights_jan_2020.columns))

flights = pd.concat([flights_jan_2019, flights_jan_2020])
flights.reset_index(drop=True, inplace=True)
print(flights.shape)

summary = pd.DataFrame({'uniue_vals': flights.nunique(), 'missing_percent': round(flights.isna().sum()*100/flights.count(), 2), 'data_type': flights.dtypes})
summary

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,6))
dep = sns.countplot(x=flights['DEP_DEL15'], hue=flights['YEAR'], ax=ax0)
dep.set_title('Depatures')
dep.set_xlabel('Labels')
dep.set_ylabel('Freq')
arr = sns.countplot(x=flights['ARR_DEL15'], hue=flights['YEAR'], ax=ax1)
arr.set_title('Arrivals')
arr.set_xlabel('Labels')
arr.set_ylabel('Freq')
plt.show()

week = flights[['DAY_OF_WEEK', 'ARR_DEL15']].groupby('DAY_OF_WEEK').sum()
week['PERCENT'] = week['ARR_DEL15']/(week['ARR_DEL15'].sum())*100
plt.figure(figsize=(7, 6))
sns.barplot(x=week.index, y=week['ARR_DEL15']).set(title='Delayed flights by day of week', xlabel='Day of Week', ylabel='Freq')
for i, v in enumerate(week['PERCENT']):
    plt.text(week.index[i]-1.25, v+1000, str('{:.1f}%'.format(v)))
plt.show()

month = flights[['DAY_OF_MONTH', 'ARR_DEL15']].groupby('DAY_OF_MONTH').sum()
month['PERCENT'] = month['ARR_DEL15']/(month['ARR_DEL15'].sum())*100
plt.figure(figsize=(25, 8))
sns.barplot(x=month.index, y=month['ARR_DEL15']).set(title='Delayed flights by day of month', xlabel='Day of Month', ylabel='Freq')
for i, v in enumerate(month['PERCENT']):
    plt.text(month.index[i]-1.25, v+250, str('{:.1f}%'.format(v)))
plt.show()

origin_later = flights[['ORIGIN', 'DEP_DEL15']].groupby('ORIGIN').sum().sort_values(by='DEP_DEL15', ascending=False)
origin_later['PERCENT'] = origin_later['DEP_DEL15']/(origin_later['DEP_DEL15'].sum())*100
origin_later.head()

dest_later = flights[['DEST', 'ARR_DEL15']].groupby('DEST').sum().sort_values(by='ARR_DEL15', ascending=False)
dest_later['PERCENT'] = dest_later['ARR_DEL15']/(dest_later['ARR_DEL15'].sum())*100
dest_later.head()

carrier = flights[['OP_UNIQUE_CARRIER', 'ARR_DEL15']].groupby('OP_UNIQUE_CARRIER').sum().sort_values(by='ARR_DEL15', ascending=False)
plt.figure(figsize=(14, 8))
sns.barplot(x=carrier.index, y=carrier['ARR_DEL15']).set(title='Delayed flights by Carrier', xlabel='Carrier Code', ylabel='Freq')
plt.show()

plt.figure(figsize=(15, 8))
sns.histplot(x=flights['DISTANCE'], hue=flights['ARR_DEL15'], bins=50, palette=["k", "r"]).set(title='Delayed flights by distance flew', xlabel='Distance', ylabel='Freq')
plt.show()

# NaN count in delayed arrivals
flights['ARR_DEL15'].isnull().sum() == flights['CANCELLED'].sum() + flights['DIVERTED'].sum()

# It seems target values are set to NaN, which flights cancelled and diverted. So for those, we assume that it arrived in late
flights.loc[flights['CANCELLED'] == 1, 'ARR_DEL15'] = 1
flights.loc[flights['DIVERTED'] == 1, 'ARR_DEL15'] = 1
flights.drop(columns=['CANCELLED', 'DIVERTED'], inplace=True)

# Since 'Unnamed: 21' column dose not carry any date, drop that column
flights.drop(columns=['Unnamed: 21'], inplace=True)

# Since all other missing values are less that 3% and the dataset is quite big, drop flights with missing data
flights = flights.dropna()
flights.reset_index(drop=True)

print(flights.shape)
flights.isnull().sum()

numeric_flights = flights.select_dtypes(include=['number'])

plt.figure(figsize = (14, 10))
sns.heatmap(numeric_flights.corr(), annot = True, cmap = 'coolwarm')
plt.show()

flights[['DEP_TIME', 'DEP_TIME_BLK']]

# It seems some of the time blocks are incorrect and it has one block for 0001-0559. Therefore, recreating time blocks for departures and creating time blocks for arrivals
blocks = []
for hour in range(0,24):
    hour_part = ('%02d' %(hour))
    blocks.append(hour_part + '00-' + hour_part + '59')

def get_time_blk(time):
    hour = str('%04d' %(time))[:2]
    time_block = None
    for block in blocks:
        if block.startswith(hour):
            time_block = block
            break
    if time_block == None and str(time) == '2400.0':
        time_block = '0000-0059'
    return time_block

flights['ARR_TIME_BLK'] = flights.ARR_TIME.apply(get_time_blk)
flights['DEP_TIME_BLK'] = flights.DEP_TIME.apply(get_time_blk)

# Rechecking the accuracy of time and time block columns
flights[['DEP_TIME', 'DEP_TIME_BLK', 'ARR_TIME', 'ARR_TIME_BLK']]

# Dropping DEP_TIME and ARR_TIME, since TIME and TIME_BLK columns will strongly correlated each other
flights.drop(columns=['DEP_TIME', 'ARR_TIME'], inplace=True)

(flights['OP_UNIQUE_CARRIER'] == flights['OP_CARRIER']).value_counts()

# Dropping OP_CARRIER, since OP_UNIQUE_CARRIER and OP_CARRIER columns are same
flights.drop(columns = ['OP_CARRIER', 'OP_CARRIER_AIRLINE_ID'], inplace=True)

flights['ORIGIN_AIRPORT_SEQ_ID'] = flights['ORIGIN_AIRPORT_SEQ_ID'].apply(str)
flights['ORIGIN_SEQ_ID'] = flights['ORIGIN_AIRPORT_SEQ_ID'].str[-2:]

flights['DEST_AIRPORT_SEQ_ID'] = flights['DEST_AIRPORT_SEQ_ID'].apply(str)
flights['DEST_SEQ_ID'] = flights['DEST_AIRPORT_SEQ_ID'].str[-2:]

flights.drop(columns=['ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID'], inplace=True)

cat_var = flights.columns.values.tolist()
cat_var.remove('DISTANCE')
for col in cat_var:
    flights[col] = flights[col].astype('category')

flights.info()

# Random undersampling of whole dataset
flights_us = flights.sample(frac=0.1, random_state=42, ignore_index=True)
print('Whole dataset: ', flights.shape)
print('Reduced dataset: ', flights_us.shape)

# Generating feature matrix and target vector
flights_X = flights_us.drop(columns=['ARR_DEL15'])
flights_y = flights_us['ARR_DEL15']
print(flights_X.shape)
print(flights_y.shape)

le = LabelEncoder()
cat_var.remove('ARR_DEL15')
for col in cat_var:
    flights_X[col] = le.fit_transform(flights_X[[col]])

flights_X.head() 

lb = LabelBinarizer()
flights_y = lb.fit_transform(flights_y).reshape((-1,))

# Splitting the data into train and test datasets (70% & 30%)
X_train, X_test, y_train, y_test = train_test_split(flights_X, flights_y, test_size=0.3, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Since the dataset is imbalanced, let's use RandomOverSampler method to generate more examples of class 1
ros = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
for dataset in [X_train_ros, y_train_ros, X_test, y_test]:
    print(dataset.shape)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,5))
unbal = sns.countplot(x=y_train, ax=ax0).set(title='Unbalance dataset')
bal = sns.countplot(x=y_train_ros, ax=ax1).set(title='Balance dataset')
plt.show()

# Instantiate the machine learning classifiers
log = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
sgd = SGDClassifier(max_iter=1000, n_jobs=-1, random_state=42)
dtc = DecisionTreeClassifier(random_state=42)
rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
svc = SVC(random_state=42)
gbc = GradientBoostingClassifier(random_state=42)
xgb = XGBClassifier(n_jobs=-1, random_state=42)

table_index = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Initail model evaluation
initial_models = [log, sgd, dtc, rfc, svc, gbc, xgb]
initial_model_scores = {}

def initial_model_eval(X, y):
    for model in initial_models:
        model.fit(X, y)
        y_pred = model.predict(X)
        initial_model_scores[type(model).__name__] = [accuracy_score(y, y_pred), precision_score(y, y_pred), recall_score(y, y_pred), f1_score(y, y_pred)]
    initial_table = pd.DataFrame(initial_model_scores, index=table_index)
    initial_table['BestScore'] = initial_table.idxmax(axis=1)
    return initial_table

initial_model_eval(X_train_ros, y_train_ros)

# Define dictionary with performance metrics
scoring = {'accuracy':make_scorer(accuracy_score), 'precision':make_scorer(precision_score), 'recall':make_scorer(recall_score), 'f1_score':make_scorer(f1_score)}

# Cross validate model evaluation
cv_models = [log, sgd, dtc, rfc, svc, gbc, xgb]
cv_model_scores = {}

def cv_model_eval(X, y, folds):
    for model in cv_models:
        scores = cross_validate(model, X, y, cv=folds, scoring=scoring, n_jobs=-1)
        cv_model_scores[type(model).__name__] = [scores['test_accuracy'].mean(), scores['test_precision'].mean(), scores['test_recall'].mean(), scores['test_f1_score'].mean()]
    cv_table = pd.DataFrame(cv_model_scores, index=table_index)
    cv_table['BestScore'] = cv_table.idxmax(axis=1)
    return cv_table

cv_model_eval(X_train_ros, y_train_ros, 5)

# Hyperparameter tuning with grid search cross validation
parameters = {'n_estimators': [100, 200, 500],
              'max_depth': [20, 40, 60],
              'max_features': [2, 3]}

gs_rfc = GridSearchCV(estimator=rfc, param_grid=parameters, cv=3, n_jobs=-1, verbose=2)
gs_rfc.fit(X_train_ros, y_train_ros)
gs_rfc.best_params_

best_rfc = gs_rfc.best_estimator_
best_pred_tr = best_rfc.predict(X_train_ros)
print('Classification report on training set')
print(classification_report(y_train_ros, best_pred_tr))

best_pred_te = best_rfc.predict(X_test)
print('Classification report on testing set')
print(classification_report(y_test, best_pred_te))

con_mat = confusion_matrix(y_test, best_pred_te)
con = sns.heatmap(con_mat, center=True, annot=True, cmap='Blues', fmt='g')
con.set_title('Confusion Matrix', fontdict={'fontsize':14}, pad=14)
con.set_ylabel('Actual')
con.set_xlabel('Predicted')
plt.show()

features = X_train.columns
importances = best_rfc.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()