import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.preprocessing.data_pipeline import preprocess_flight_data

# Use environment variable if available; fallback to local relative path
data_dir = os.environ.get("DATA_DIR", "data/raw")

# Chooses the most recent of two available datasets for model training.
# This helps the model generalize better by focusing on up-to-date patterns, reducing the risk of
# overfitting to outdated data.
data_source = os.path.join(data_dir, "jan_2020_ontime.csv")


# --------------------------------
# Model Training Function
# --------------------------------
# Uses a Random Forest Classifier, selected based on superior performance in prior academic
# work (see notebooks and reports directories).
# The trained model is saved as a pickle file in the data/model directory for future inference
# and reuse.

def train_model():
    print("🔃 Loading data...")
    df = pd.read_csv(data_source)

    print("🧼 Preprocessing...")
    X, y, encoder = preprocess_flight_data(df)

    print("📊 Splitting train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    print("🎯 Balancing training set with RandomOverSampler...")
    ros = RandomOverSampler(sampling_strategy='minority', random_state=42)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

    print("🔠 Encoding categorical features with per-column encoders...")
    cat_cols = X_train_ros.select_dtypes(include=['object', 'category']).columns
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        X_train_ros[col] = le.fit_transform(X_train_ros[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le

    print("🔍 Performing GridSearchCV...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [20, 40],
        'max_features': [3, 'sqrt']
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train_ros, y_train_ros)
    best_model = grid_search.best_estimator_

    print("✨ Best Parameters:", grid_search.best_params_)

    print("📈 Evaluating on test set...")
    # Apply same encoders to test set categorical columns
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("💡 Feature Importance:")
    importances = best_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X_train_ros.columns,
        'Importance (%)': importances * 100
    }).sort_values(by='Importance (%)', ascending=False)
    importance_df['Importance (%)'] = importance_df['Importance (%)'].map(lambda x: f"{x:.2f}%")
    print(importance_df)

    print("💾 Saving model and artifacts...")
    joblib.dump(best_model, "data/model/model.pkl")
    joblib.dump(encoders, "data/model/encoders.pkl")
    joblib.dump(list(X.columns), "data/model/feature_columns.pkl")

    print("✅ Training complete and model saved!")


if __name__ == "__main__":
    train_model()