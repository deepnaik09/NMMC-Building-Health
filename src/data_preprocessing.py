"""
data_preprocessing.py
---------------------
Feature engineering utilities for building dataset.
Used in both training (train_model.py) and Streamlit app.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def engineer_features(df: pd.DataFrame, current_year: int = None) -> pd.DataFrame:
    """Create derived numeric features for ML model."""
    df = df.copy()
    if current_year is None:
        current_year = datetime.now().year

    df["age"] = current_year - df["construction_year"].astype(int)
    df["overbuilt_pct"] = (df["built_up_area_sqm"] - df["sanctioned_built_up_area"]) / (
        df["sanctioned_built_up_area"] + 1e-6
    )
    df["overbuilt_flag"] = (df["overbuilt_pct"] > 0.02).astype(int)
    df["no_oc_flag"] = (df["occupancy_certificate"] == 0).astype(int)
    df["no_fire_flag"] = (df["fire_noc"] == 0).astype(int)
    df["struc_failed_flag"] = (df["structural_audit_result"] == 0).astype(int)
    df["people_per_flat"] = df["no_of_people"] / (df["no_of_flats"] + 1e-6)
    df["sqm_per_floor"] = df["built_up_area_sqm"] / (df["no_of_floors"] + 1e-6)
    df["damage_per_floor"] = df["damage_count"] / (df["no_of_floors"] + 1e-6)
    return df


def get_feature_columns() -> list:
    """Return the list of model features used for training."""
    return [
        "age",
        "built_up_area_sqm",
        "sanctioned_built_up_area",
        "overbuilt_pct",
        "overbuilt_flag",
        "no_oc_flag",
        "no_fire_flag",
        "struc_failed_flag",
        "damage_count",
        "no_of_floors",
        "no_of_flats",
        "no_of_people",
        "maintenance_per_month",
        "people_per_flat",
        "sqm_per_floor",
        "damage_per_floor",
    ]
