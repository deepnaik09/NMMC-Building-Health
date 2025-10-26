"""
model_utils.py
--------------
Helpers to load trained model, predict, and compute final rating.
"""

import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from .data_preprocessing import engineer_features
from .rules_engine import apply_nmmc_rules


def load_model(model_path="models/rf_model.joblib", feature_path="models/feature_list.json"):
    model = joblib.load(model_path)
    with open(feature_path, "r") as f:
        features = json.load(f)
    return model, features


def predict_building(df_input: pd.DataFrame, model, feature_list):
    """Run ML prediction for building rating (0–10)."""
    df_features = engineer_features(df_input)
    X = df_features[feature_list].astype(float)
    preds = model.predict(X)
    return np.clip(preds, 0, 10)


def compute_final_rating(df_input: pd.DataFrame, model, feature_list):
    """Full pipeline: ML prediction + rule-based compliance penalties."""
    preds = predict_building(df_input, model, feature_list)
    final_rows = []
    for i, row in df_input.iterrows():
        comp_score, flags = apply_nmmc_rules(row)
        comp_norm = comp_score / 10.0  # convert 0–100 to 0–10
        final_rating = round(0.45 * preds[i] + 0.55 * comp_norm, 2)
        maint = float(row.get("maintenance_per_month", 0))
        maint_est = round(maint * 12 * (1 + max(0, (80 - comp_score) / 100)), 0)
        final_rows.append({
            "sr_no": i + 1,
            "building_id": row.get("building_id", f"BLDG_{i+1}"),
            "final_rating_out_of_10": final_rating,
            "nmmc_compliance_score_100": round(comp_score, 1),
            "maintenance_estimate_inr_ann": maint_est,
            "remarks": ", ".join(flags) if flags else "OK"
        })
    return pd.DataFrame(final_rows)
