
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle

# ── Load artifacts (run once at startup) ──
with open("scaler.pkl", "rb") as f:      scaler = pickle.load(f)
with open("encoders.pkl", "rb") as f:    le_dict = pickle.load(f)
with open("feature_cols.pkl", "rb") as f: feature_cols = pickle.load(f)

class StudentNet(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x): return self.network(x)

input_dim = len(feature_cols)
model_bin   = StudentNet(input_dim, 2);   model_bin.load_state_dict(torch.load("model_pass_fail.pth", map_location="cpu")); model_bin.eval()
model_grade = StudentNet(input_dim, len(le_dict["Grade"].classes_)); model_grade.load_state_dict(torch.load("model_grade.pth", map_location="cpu")); model_grade.eval()

def predict_student(data: dict) -> dict:
    row = pd.DataFrame([data])
    sem = ["Sem1_Marks","Sem2_Marks","Sem3_Marks","Sem4_Marks"]
    row["Mark_Consistency"]  = row[sem].std(axis=1) if all(s in row for s in sem) else 0
    row["Improvement_Trend"] = row["Sem4_Marks"] - row["Sem1_Marks"] if all(s in row for s in sem) else 0
    row["Engagement_Score"]  = (row.get("StudyHours",0)*0.3 + row.get("Attendance",0)*0.3 +
                                row.get("OnlineCourses",0)*0.2 + row.get("Discussions",0)*0.2)
    if "Gender" in row and "Gender" in le_dict:
        try: row["Gender"] = le_dict["Gender"].transform(row["Gender"])
        except: row["Gender"] = 0
    X = scaler.transform(row[feature_cols].fillna(0).values)
    t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        pf_prob    = torch.softmax(model_bin(t),   dim=1)[0].numpy()
        grade_prob = torch.softmax(model_grade(t), dim=1)[0].numpy()
    return {
        "pass_fail":   "Pass" if np.argmax(pf_prob) == 1 else "Fail",
        "pass_prob":   round(float(pf_prob[1]), 4),
        "grade":       le_dict["Grade"].inverse_transform([np.argmax(grade_prob)])[0],
        "grade_probs": {le_dict["Grade"].inverse_transform([i])[0]: round(float(p),4) for i,p in enumerate(grade_prob)},
    }
