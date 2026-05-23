import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

FEATURE_COLUMNS = [
    "duration",
    "focus",
    "day_of_week",
    "hour_sin",
    "hour_cos",
    "task_type",
    "user_id",
]

def load_raw_data(file_path):
    df = pd.read_excel(file_path, usecols=range(8))
    print(f"[preprocessing] Loaded {len(df)} rows.")
    return df

def rename_columns(df):
    df.columns = [
        "timestamp",
        "task_type",
        "duration",
        "start_time",
        "end_time",
        "success",
        "focus",
        "user_id"
    ]
    return df

def clean_data(df):
    df = df.dropna()
    df["duration"]   = pd.to_numeric(df["duration"],   errors="coerce")
    df["focus"]      = pd.to_numeric(df["focus"],      errors="coerce")
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"]   = pd.to_datetime(df["end_time"],   errors="coerce")
    df = df.dropna(subset=["duration", "focus", "start_time", "end_time"])
    df["success"] = df["success"].map({"Yes": 1, "No": 0})
    df = df.dropna(subset=["success"])
    df["success"] = df["success"].astype(int)
    # Reference patch — normalize text so metadata values match user input
    df["task_type"] = df["task_type"].str.strip().str.title()
    df["user_id"]   = df["user_id"].str.strip().str.upper()
    print(f"[preprocessing] After cleaning: {len(df)} rows. "
          f"Success rate: {df['success'].mean():.1%}")
    return df

def add_time_features(df):
    df["day_of_week"]  = df["start_time"].dt.dayofweek
    df["hour_decimal"] = (
        df["start_time"].dt.hour +
        df["start_time"].dt.minute / 60
    )
    return df


def add_cyclical_features(df):
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_decimal"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_decimal"] / 24)
    return df

def prepare_training_data(file_path):
    df = load_raw_data(file_path)
    df = rename_columns(df)
    df = clean_data(df)
    df = add_time_features(df)
    df = add_cyclical_features(df)
    X = df[FEATURE_COLUMNS]
    y = df["success"]
    print(f"[preprocessing] Feature matrix: {X.shape}")
    return X, y

def prepare_focus_data(file_path):
    df = load_raw_data(file_path)
    df = rename_columns(df)
    df = clean_data(df)
    df = add_time_features(df)
    df = add_cyclical_features(df)
    X       = df[FEATURE_COLUMNS]
    y_focus = df["focus"]
    print(f"[preprocessing] Focus target — mean: {y_focus.mean():.2f}, "
          f"std: {y_focus.std():.2f}")
    return X, y_focus


def build_preprocessor():
    categorical_features = ["task_type", "user_id"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough"
    )
    return preprocessor

def create_candidate_feature_row(user_id, task_type, duration, start_time):
    hour_decimal = start_time.hour + start_time.minute / 60.0
    day_of_week  = start_time.weekday()
    hour_sin     = np.sin(2 * np.pi * hour_decimal / 24)
    hour_cos     = np.cos(2 * np.pi * hour_decimal / 24)
    row = {
        "duration":    duration,
        "focus":       3.0,
        "day_of_week": day_of_week,
        "hour_sin":    hour_sin,
        "hour_cos":    hour_cos,
        "task_type":   task_type,
        "user_id":     user_id,
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]
