from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
sys.path.append('/opt/airflow/app')

# Import actual function
from model.train_model import train_model

# Dummy preprocessing step for logging (actual preprocessing is handled in train_model)
def run_preprocessing():
    print("No-op preprocessing step (handled in train_model.py)")
    
# Training Task Wrapper
def run_training():
    train_model()

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

with DAG(
    dag_id="flight_delay_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["ml", "flights"]
) as dag:

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=run_preprocessing
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=run_training
    )

    preprocess >> train