import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from AI_Project_Preprocessing import (
    prepare_training_data,
    prepare_focus_data,
    build_preprocessor,
)

RANDOM_STATE     = 42
TEST_SIZE        = 0.2
MODEL_PATH       = "model.joblib"
FOCUS_MODEL_PATH = "focus_model.joblib"
METADATA_PATH    = "metadata.joblib"


def split_dataset(X, y):
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def create_model_pipelines():
    logistic_pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ])
    rf_pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])
    return {
        "LogisticRegression": logistic_pipeline,
        "RandomForest":       rf_pipeline
    }

def evaluate(model, X_test, y_test, name):
    predictions   = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    acc  = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, zero_division=0)
    rec  = recall_score(y_test, predictions, zero_division=0)
    f1   = f1_score(y_test, predictions, zero_division=0)
    auc  = roc_auc_score(y_test, probabilities)
    print(f"\n--- {name} Results ---")
    print(classification_report(y_test, predictions))
    print("ROC-AUC:", round(auc, 3))
    return {"accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "roc_auc": auc}


def train_and_evaluate(X_train, X_test, y_train, y_test):
    pipelines     = create_model_pipelines()
    trained_models = {}
    results        = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        results[name]        = evaluate(pipeline, X_test, y_test, name)
    return trained_models, results


def select_final_model(trained_models, results):
    final_name  = max(results, key=lambda name: results[name]["roc_auc"])
    final_model = trained_models[final_name]
    lr_auc      = results["LogisticRegression"]["roc_auc"]
    rf_auc      = results["RandomForest"]["roc_auc"]
    print(f"\n  Logistic Regression ROC-AUC : {lr_auc:.3f}")
    print(f"  Random Forest ROC-AUC       : {rf_auc:.3f}")
    print(f"\n  ✓ Selected model: {final_name}")
    return final_model, final_name


def train_focus_regressor(X, y_focus):
    focus_pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE
        ))
    ])
    kf             = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    neg_mse_scores = cross_val_score(
        focus_pipeline, X, y_focus,
        cv=kf, scoring="neg_mean_squared_error"
    )
    rmse = (-neg_mse_scores) ** 0.5
    print(f"\n[train_model] Focus Regressor 5-Fold CV RMSE: "
          f"{rmse.mean():.3f} ± {rmse.std():.3f}  (scale: 1–5)")
    focus_pipeline.fit(X, y_focus)
    print(f"[train_model] Focus regressor trained on {len(X)} rows.")
    return focus_pipeline


def build_training_metadata(final_model_name, X):
    metadata = {
        "model_name":              final_model_name,
        "search_start_hour":       6,
        "search_end_hour":         22,
        "interval_minutes":        30,
        # Reference patch — So omitted these two
        "task_types":  sorted(X["task_type"].unique().tolist()),
        "user_ids":    sorted(X["user_id"].unique().tolist()),
    }
    return metadata

def save_outputs(model, focus_model, metadata):
    joblib.dump(model,       MODEL_PATH)
    joblib.dump(focus_model, FOCUS_MODEL_PATH)
    joblib.dump(metadata,    METADATA_PATH)
    print("\n[train_model] model.joblib, focus_model.joblib, "
          "metadata.joblib saved.")

def run_training_pipeline(file_path):
    print("\n" + "="*50)
    print("  STEP 1: LOAD AND PREPARE DATA")
    print("="*50)
    X, y = prepare_training_data(file_path)

    print("\n" + "="*50)
    print("  STEP 2: SPLIT DATA")
    print("="*50)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print("\n" + "="*50)
    print("  STEP 3: TRAIN AND EVALUATE CLASSIFIERS")
    print("="*50)
    trained_models, results = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\n" + "="*50)
    print("  STEP 4: SELECT FINAL CLASSIFIER")
    print("="*50)
    final_model, final_name = select_final_model(trained_models, results)

    print("\n" + "="*50)
    print("  STEP 5: TRAIN FOCUS REGRESSOR")
    print("="*50)
    _, y_focus   = prepare_focus_data(file_path)
    focus_model  = train_focus_regressor(X, y_focus)

    print("\n" + "="*50)
    print("  STEP 6: SAVE ARTIFACTS")
    print("="*50)
    metadata = build_training_metadata(final_name, X)
    save_outputs(final_model, focus_model, metadata)

    print("\n[train_model] Training complete.")
    return final_model, focus_model, metadata


if __name__ == "__main__":
    FILE_PATH = "AI Model (Responses).xlsx"
    print("Starting training...")
    run_training_pipeline(FILE_PATH)
    print("Done.")
