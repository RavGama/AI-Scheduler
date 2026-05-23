import sys
import os
import joblib

from datetime import datetime, timedelta

from AI_Project_Train_Model import run_training_pipeline
from AI_Project_Agent import (
    load_system_artifacts,
    recommend_times,
    display_results,
    log_results,
    get_confidence_label,
)
from AI_Project_Preprocessing import create_candidate_feature_row

FILE_PATH        = "AI Model (Responses).xlsx"
MODEL_PATH       = "model.joblib"
FOCUS_MODEL_PATH = "focus_model.joblib"
METADATA_PATH    = "metadata.joblib"
LOG_PATH         = "recommendation_log.csv"


def get_user_inputs(metadata):
    print("\n" + "="*50)
    print("  SCHEDULING ASSISTANT")
    print("="*50)
    print("  Answer the questions below to get your")
    print("  personalized time recommendations.\n")

    print(f"  Known users: {metadata['user_ids']}")
    while True:
        user_id = input("\n  Enter your User ID: ").strip().upper()
        if user_id:
            break
        print("  Please enter a User ID.")

    print(f"\n  Valid task types: {metadata['task_types']}")
    while True:
        task_type = input("  Enter task type: ").strip().title()
        if task_type in metadata["task_types"]:
            break
        print(f"  Invalid. Choose from: {metadata['task_types']}")

    while True:
        try:
            duration = float(input("  How long will the task take (minutes)? ").strip())
            if duration > 0:
                break
            print("  Please enter a number greater than 0.")
        except ValueError:
            print("  Please enter a valid number.")

    while True:
        try:
            days_ahead = int(input("  How many days ahead to search? (1-7): ").strip())
            if 1 <= days_ahead <= 7:
                break
            print("  Please enter a number between 1 and 7.")
        except ValueError:
            print("  Please enter a whole number.")

    print("\n  Would you like to check a specific start time?")
    check_specific = input("  Enter yes or no: ").strip().lower()
    specific_time  = None
    if check_specific in ("yes", "y"):
        while True:
            try:
                time_str = input(
                    "  Enter start time in military time (e.g. 9:00 or 14:30): "
                ).strip()
                for fmt in ("%I:%M %p", "%H:%M"):
                    try:
                        parsed        = datetime.strptime(time_str, fmt)
                        specific_time = parsed
                        break
                    except ValueError:
                        continue
                if specific_time:
                    break
                print("  Format not recognised. Try '9:00 AM' or '14:30'.")
            except Exception:
                print("  Please try again.")

    return {
        "user_id":       user_id,
        "task_type":     task_type,
        "duration":      duration,
        "days_ahead":    days_ahead,
        "specific_time": specific_time,
    }


def run_training():
    print("\n" + "█"*50)
    print("  PHASE 1: TRAINING THE MODELS")
    print("█"*50)
    _model, _focus_model, metadata = run_training_pipeline(FILE_PATH)
    print("\n✅ Training complete.")
    return metadata


def run_agent_demo(metadata):
    print("\n" + "█"*50)
    print("  PHASE 2: SCHEDULING AGENT")
    print("█"*50)

    model, focus_model, metadata = load_system_artifacts()
    inputs        = get_user_inputs(metadata)
    specific_time = inputs.pop("specific_time")

    if specific_time:
        now     = datetime.now()
        slot_dt = now.replace(
            hour=specific_time.hour,
            minute=specific_time.minute,
            second=0, microsecond=0,
        )
        row          = create_candidate_feature_row(
            inputs["user_id"], inputs["task_type"], inputs["duration"], slot_dt
        )
        prob         = model.predict_proba(row)[0][1]
        focus_score  = 3.0
        if focus_model is not None:
            focus_score = float(focus_model.predict(row)[0])
            focus_score = max(1.0, min(5.0, focus_score))
        composite = 0.65 * prob + 0.35 * (focus_score / 5.0)
        label     = get_confidence_label(composite)

        print("\n" + "="*55)
        print("  YOUR CHOSEN TIME")
        print("="*55)
        try:
            start_str = slot_dt.strftime("%-I:%M %p")
            end_str   = (slot_dt + timedelta(minutes=60)).strftime("%-I:%M %p")
        except ValueError:
            start_str = slot_dt.strftime("%#I:%M %p")
            end_str   = (slot_dt + timedelta(minutes=60)).strftime("%#I:%M %p")
        print(f"\n    {start_str} – {end_str}")
        print(f"    Success Probability : {prob * 100:.1f}%")
        print(f"    Predicted Focus     : {focus_score:.1f} / 5.0")
        print(f"    Composite Score     : {composite * 100:.1f}%")
        print(f"     Confidence         : {label}")
        print("="*55)

    print("\n  Finding the best times for you...\n")
    recommendations = recommend_times(model, focus_model, metadata, **inputs)
    display_results(recommendations)
    log_results(recommendations)


def main():
    print("=" * 50)
    print("  AI Scheduling System")
    print("  Insight Strategy Group")
    print("=" * 50)
    print("\n  Usage:")
    print("    python AI_Project_Run.py              (train + run)")
    print("    python AI_Project_Run.py --skip-train (skip training)")

    skip_train = "--skip-train" in sys.argv

    if skip_train:
        for path, label in [(MODEL_PATH, "model.joblib"),
                             (METADATA_PATH, "metadata.joblib")]:
            if not os.path.exists(path):
                print(f"\n  '{label}' not found.")
                print("  Run without --skip-train first to train the models.")
                sys.exit(1)
        metadata = joblib.load(METADATA_PATH)
        print("\n  Loaded existing model artifacts.")
    else:
        metadata = run_training()

    run_agent_demo(metadata)
    print("\n  Done.")


if __name__ == "__main__":
    main()
