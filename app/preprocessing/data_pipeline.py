import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

# --------------------------------
# Recreate Time Blocks
# --------------------------------
# Creates hourly time block labels for 24 hours. Existing blocks were found to be inaccurate and
# inconsistently defined, with some spanning multiple hours.

def generate_time_blocks():
    return [f'{h:02d}00-{h:02d}59' for h in range(24)]

def get_time_block(time_val, blocks):
    try:
        hour = str(f'{int(time_val):04d}')[:2]
        for block in blocks:
            if block.startswith(hour):
                return block
        if str(time_val) == '2400':
            return '0000-0059'
    except:
        return 'Unknown'
    return 'Unknown'


# --------------------------------
# Main Preprocessing Function
# --------------------------------
# Cleans and preprocesses the flight delay dataset, returning a tuple:
#   - X_encoded (pd.DataFrame): Feature matrix after encoding
#   - y (pd.Series): Target vector
#   - label_encoder (LabelEncoder): Fitted encoder for future use

def preprocess_flight_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    df = df.copy()

    # Fill delayed arrivals (target) for cancelled/diverted
    df.loc[df['CANCELLED'] == 1, 'ARR_DEL15'] = 1
    df.loc[df['DIVERTED'] == 1, 'ARR_DEL15'] = 1

    # Drop unnecessary columns
    df.drop(columns=['CANCELLED', 'DIVERTED'], inplace=True, errors='ignore')
    df.drop(columns=['Unnamed: 21'], inplace=True, errors='ignore')

    # Drop nulls
    df.dropna(inplace=True)

    # Create time blocks for DEP and ARR
    blocks = generate_time_blocks()
    df['ARR_TIME_BLK'] = df['ARR_TIME'].apply(lambda t: get_time_block(t, blocks))
    df['DEP_TIME_BLK'] = df['DEP_TIME'].apply(lambda t: get_time_block(t, blocks))
    df.drop(columns=['DEP_TIME', 'ARR_TIME'], inplace=True)

    # Remove duplicate columns
    if 'OP_CARRIER' in df.columns and 'OP_UNIQUE_CARRIER' in df.columns:
        df.drop(columns=['OP_CARRIER'], inplace=True)

    # Drop ID columns
    df.drop(columns=['OP_CARRIER_AIRLINE_ID'], inplace=True, errors='ignore')
    df['ORIGIN_AIRPORT_SEQ_ID'] = df['ORIGIN_AIRPORT_SEQ_ID'].astype(str)
    df['ORIGIN_SEQ_ID'] = df['ORIGIN_AIRPORT_SEQ_ID'].str[-2:]
    df['DEST_AIRPORT_SEQ_ID'] = df['DEST_AIRPORT_SEQ_ID'].astype(str)
    df['DEST_SEQ_ID'] = df['DEST_AIRPORT_SEQ_ID'].str[-2:]
    df.drop(columns=['ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID',
                     'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID'], inplace=True, errors='ignore')

    # Random downsample of data set
    df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)

    # Split features/target
    X = df.drop(columns=['ARR_DEL15'])
    y = df['ARR_DEL15']

    # Encode categorical features
    label_encoder = LabelEncoder()
    cat_cols = X.select_dtypes(['category', 'object']).columns
    for col in cat_cols:
        X[col] = label_encoder.fit_transform(X[col].astype(str))

    return X, y, label_encoder