# AI Personalized Scheduling System

A machine learning system that predicts optimal task scheduling 
windows based on individual behavioral patterns, achieving 
0.84 ROC-AUC on held-out data.

## Overview
Generic scheduling tools ignore individual behavior. This system 
learns from a user's historical task logs to predict when they 
are most likely to successfully complete a task and maintain 
focus — then ranks candidate time windows accordingly.

## Architecture
The system uses a dual-model approach:
- **Success Classifier** — Random Forest classifier predicting 
  the probability a task will be completed successfully
- **Focus Regressor** — Random Forest regressor predicting 
  expected focus score (1–5 scale)

Both scores are blended into a composite ranking score to 
surface the optimal time windows for each user.

## Results
| Model | ROC-AUC |
|---|---|
| Logistic Regression | baseline |
| Random Forest | 0.84 |

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train models and run the scheduling agent
python src/AI_Project_Run.py

### 3. Skip retraining if models are already saved
python src/AI_Project_Run.py --skip-train

## Input Data
The system expects an Excel file with the following columns:
- timestamp, task_type, duration, start_time, end_time, 
  success, focus, user_id

## Output
- `model.joblib` — trained success classifier
- `focus_model.joblib` — trained focus regressor  
- `metadata.joblib` — task types, user IDs, search parameters
- `recommendation_log.csv` — logged recommendations per session

## Tech Stack
Python, scikit-learn, pandas, NumPy, joblib
