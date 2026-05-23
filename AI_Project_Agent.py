import os
import joblib
import pandas as pd

from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from AI_Project_Preprocessing import create_candidate_feature_row

MODEL_PATH       = "model.joblib"
FOCUS_MODEL_PATH = "focus_model.joblib"
METADATA_PATH    = "metadata.joblib"
LOG_FILE         = "recommendation_log.csv"

SUCCESS_WEIGHT = 0.65
FOCUS_WEIGHT   = 0.35

@dataclass
class Recommendation:
    user_id:             str
    task_type:           str
    duration:            float
    start_time:          datetime
    end_time:            datetime
    probability:         float
    predicted_focus:     float
    composite_score:     float
    confidence:          str

def load_system_artifacts():
    model    = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    print(f"[agent] Classifier loaded.")

    focus_model = None
    if os.path.exists(FOCUS_MODEL_PATH):
        focus_model = joblib.load(FOCUS_MODEL_PATH)
        print(f"[agent] Focus regressor loaded.")
    else:
        print(f"[agent] Focus model not found — ranking by success only.")

    print(f"[agent] {len(metadata['user_ids'])} users, "
          f"{len(metadata['task_types'])} task types.")
    return model, focus_model, metadata

def get_confidence_label(prob):
    if prob >= 0.80:
        return " High Confidence"
    elif prob >= 0.60:
        return " Moderate Confidence"
    else:
        return " Low Confidence"

def generate_time_windows(days_ahead, start_hour=6, end_hour=22, interval=30):
    now     = datetime.now()
    windows = []
    for day_offset in range(1, days_ahead + 1):
        base_day     = now + timedelta(days=day_offset)
        current_time = base_day.replace(hour=start_hour, minute=0,
                                        second=0, microsecond=0)
        end_time     = base_day.replace(hour=end_hour, minute=0,
                                        second=0, microsecond=0)
        while current_time <= end_time:
            windows.append(current_time)
            current_time += timedelta(minutes=interval)
    return windows

def evaluate_time_windows(model, focus_model, user_id, task_type,
                           duration, windows):
    results = []
    for start in windows:
        row          = create_candidate_feature_row(user_id, task_type,
                                                    duration, start)
        prob         = model.predict_proba(row)[0][1]
        focus_score  = 3.0
        if focus_model is not None:
            focus_score = float(focus_model.predict(row)[0])
            focus_score = max(1.0, min(5.0, focus_score))
        composite = SUCCESS_WEIGHT * prob + FOCUS_WEIGHT * (focus_score / 5.0)
        results.append((start, prob, focus_score, composite))
    return results

def recommend_times(model, focus_model, metadata, user_id, task_type,
                    duration, days_ahead=3, top_n=5):
    user_id   = str(user_id).strip().upper()
    task_type = str(task_type).strip().title()
    duration  = float(duration)

    if task_type not in metadata["task_types"]:
        raise ValueError(f"Unknown task type '{task_type}'. "
                         f"Valid: {metadata['task_types']}")
    if user_id not in metadata["user_ids"]:
        print(f"[agent]   User '{user_id}' not in training data. "
              "Predictions will not be personalised.")

    print(f"\n[agent] Searching: {task_type} | User: {user_id} | "
          f"{duration} min | {days_ahead} day(s) ahead")

    windows = generate_time_windows(
        days_ahead,
        start_hour=metadata.get("search_start_hour", 6),
        end_hour=metadata.get("search_end_hour", 22),
        interval=metadata.get("interval_minutes", 30),
    )
    print(f"[agent] Evaluating {len(windows)} candidate slots...")

    scored = evaluate_time_windows(model, focus_model, user_id,
                                    task_type, duration, windows)
    scored.sort(key=lambda x: x[3], reverse=True)

    recommendations = []
    for start, prob, focus_score, composite in scored[:top_n]:
        end   = start + timedelta(minutes=duration)
        label = get_confidence_label(composite)
        rec   = Recommendation(
            user_id=user_id,
            task_type=task_type,
            duration=duration,
            start_time=start,
            end_time=end,
            probability=prob,
            predicted_focus=focus_score,
            composite_score=composite,
            confidence=label,
        )
        recommendations.append(rec)
    return recommendations

def log_results(recommendations):
    rows = []
    for rec in recommendations:
        rows.append({
            "user_id":         rec.user_id,
            "task_type":       rec.task_type,
            "duration":        rec.duration,
            "start_time":      rec.start_time.strftime("%Y-%m-%d %H:%M"),
            "end_time":        rec.end_time.strftime("%Y-%m-%d %H:%M"),
            "probability":     f"{rec.probability:.4f}",
            "predicted_focus": f"{rec.predicted_focus:.2f}",
            "composite_score": f"{rec.composite_score:.4f}",
            "confidence":      rec.confidence,
        })
    df = pd.DataFrame(rows)
    try:
        existing = pd.read_csv(LOG_FILE)
        df = pd.concat([existing, df], ignore_index=True)
    except FileNotFoundError:
        pass
    df.to_csv(LOG_FILE, index=False)
    print(f"[agent] {len(recommendations)} recommendation(s) logged to '{LOG_FILE}'")

def display_results(recommendations):
    print("\n" + "="*60)
    print(f"    TOP {len(recommendations)} RECOMMENDED TIME WINDOWS")
    print("="*60)
    for i, rec in enumerate(recommendations, start=1):
        date_str = rec.start_time.strftime("%A, %B %d")
        try:
            start_str = rec.start_time.strftime("%-I:%M %p")
            end_str   = rec.end_time.strftime("%-I:%M %p")
        except ValueError:
            start_str = rec.start_time.strftime("%#I:%M %p")
            end_str   = rec.end_time.strftime("%#I:%M %p")
        print(f"\n  Rank #{i}")
        print(f"    {date_str}")
        print(f"    {start_str} – {end_str}")
        print(f"    Success Probability : {rec.probability * 100:.1f}%")
        print(f"    Predicted Focus     : {rec.predicted_focus:.1f} / 5.0")
        print(f"    Composite Score     : {rec.composite_score * 100:.1f}%")
        print(f"     Confidence         : {rec.confidence}")
    print("\n" + "="*60 + "\n")
