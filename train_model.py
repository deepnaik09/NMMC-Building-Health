"""
train_model.py

Generates synthetic Navi Mumbai building dataset, trains a RandomForestRegressor
to predict a preliminary building score, and saves the model + preprocessing.

Run:
    python train_model.py

Outputs:
    models/rf_model.joblib
    models/feature_list.json
"""

import os
import json
import random
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump

# --------------------------
# Settings
# --------------------------
N = 1500  # number of synthetic samples to create
CURRENT_YEAR = datetime.now().year
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(42)
np.random.seed(42)

# --------------------------
# Real Navi Mumbai localities (seed coordinates)
# Sources: Vashi, Kharghar, Panvel, Belapur coords (public sources)
# (These were looked up and used as seed centers for society coordinates.)
# --------------------------
locality_seeds = [
    {"name": "Vashi", "lat": 19.077065, "lon": 72.998993},       # source: latlong.net
    {"name": "Kharghar", "lat": 19.047321, "lon": 73.069908},    # source: latlong.net
    {"name": "Panvel", "lat": 18.990713, "lon": 73.116844},      # source: latlong.net
    {"name": "Belapur", "lat": 19.0330488, "lon": 73.0296625},   # source: geodatos
]

# --------------------------
# Helper: make synthetic sample
# --------------------------
def make_sample(i):
    seed = random.choice(locality_seeds)
    # small jitter around seed coords (within ~0.003 deg ~ 300m)
    lat = seed["lat"] + np.random.normal(scale=0.0025)
    lon = seed["lon"] + np.random.normal(scale=0.0025)

    construction_year = int(np.random.randint(1960, CURRENT_YEAR + 1))

    age = CURRENT_YEAR - construction_year

    # built up and sanctioned. built_up may exceed sanctioned
    sanctioned = np.round(np.random.uniform(400, 10000))  # sqm - value for whole building
    # built up roughly around sanctioned ± 5-25%
    built_up = sanctioned * np.random.uniform(0.85, 1.25)

    occupancy_certificate = np.random.choice([0,1], p=[0.12, 0.88])  # 1 means OC present
    fire_noc = np.random.choice([0,1], p=[0.3, 0.7])                 # 1 means Fire NOC present
    # structural audit result: 1=pass, 0=fail
    # buildings older than 30 more likely to have audit issues
    if age > 50:
        structural_audit_result = np.random.choice([0,1], p=[0.25, 0.75])
    elif age > 30:
        structural_audit_result = np.random.choice([0,1], p=[0.15, 0.85])
    else:
        structural_audit_result = np.random.choice([0,1], p=[0.05, 0.95])

    damage_count = int(np.random.poisson(lam=max(0.5, age/30)))
    no_of_floors = int(np.clip(np.random.poisson(lam=6), 1, 40))
    no_of_flats = int(np.clip(no_of_floors * np.random.randint(2,8), 2, 400))
    no_of_people = int(np.clip(no_of_flats * np.random.randint(1,5), 1, 2000))
    maintenance_per_month = round(np.clip((built_up/1000) * np.random.uniform(5000,15000), 2000, 200000))

    # simple synthetic true score (ground-truth) for training:
    # base score depends negatively on age, damage_count and missing documents
    base = 8.0
    base -= (age / 100) * 3.5
    base -= (damage_count * 0.4)
    # penalties for missing OC or fire NOC or failed audit
    if occupancy_certificate == 0:
        base -= 2.0
    if fire_noc == 0:
        base -= 1.2
    if structural_audit_result == 0:
        base -= 2.5
    # FSI penalty
    if built_up > sanctioned * 1.02:
        base -= 1.0
    # parking/utility noise: randomly degrade a bit
    base += np.random.normal(scale=0.8)
    score = float(np.clip(base, 0.2, 9.9))  # ground-truth label 0..10

    return {
        "building_id": f"BNM{10000 + i}",
        "construction_year": construction_year,
        "built_up_area_sqm": round(built_up, 2),
        "sanctioned_built_up_area": round(sanctioned, 2),
        "occupancy_certificate": occupancy_certificate,
        "fire_noc": fire_noc,
        "structural_audit_result": structural_audit_result,
        "damage_count": damage_count,
        "no_of_floors": no_of_floors,
        "no_of_flats": no_of_flats,
        "no_of_people": no_of_people,
        "maintenance_per_month": maintenance_per_month,
        "label_score": score
    }

# --------------------------
# Create dataset
# --------------------------
rows = [make_sample(i) for i in range(N)]
df = pd.DataFrame(rows)

# Shuffle and inspect distribution
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Samples:", df.shape)
print(df[["construction_year","label_score"]].describe())

# --------------------------
# Feature engineering for model:
# We'll create inputs ML_features. The model predicts label_score (~0-10).
# --------------------------
def engineer_features(df):
    df = df.copy()
    # Age
    df["age"] = CURRENT_YEAR - df["construction_year"]
    # overbuilt percent
    df["overbuilt_pct"] = (df["built_up_area_sqm"] - df["sanctioned_built_up_area"]) / (df["sanctioned_built_up_area"] + 1e-6)
    df["overbuilt_flag"] = (df["overbuilt_pct"] > 0.02).astype(int)
    # missing docs
    df["no_oc_flag"] = (df["occupancy_certificate"] == 0).astype(int)
    df["no_fire_flag"] = (df["fire_noc"] == 0).astype(int)
    df["struc_failed_flag"] = (df["structural_audit_result"] == 0).astype(int)
    # people per flat
    df["people_per_flat"] = df["no_of_people"] / (df["no_of_flats"] + 1e-6)
    # built_up per floor
    df["sqm_per_floor"] = df["built_up_area_sqm"] / (df["no_of_floors"] + 1e-6)
    # damage per floor
    df["damage_per_floor"] = df["damage_count"] / (df["no_of_floors"] + 1e-6)
    return df

df = engineer_features(df)

feature_cols = [
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
    "damage_per_floor"
]

X = df[feature_cols]
y = df["label_score"]

# --------------------------
# Train/test
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)

# We'll use a simple pipeline (scaler + RF)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(n_estimators=250, max_depth=12, random_state=42, n_jobs=-1))
])

print("Training model...")
pipeline.fit(X_train, y_train)

# Quick eval
from sklearn.metrics import mean_squared_error, r2_score
pred = pipeline.predict(X_test)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)
print(f"Test RMSE: {rmse:.3f}, R2: {r2:.3f}")

# Save artifacts
dump(pipeline, os.path.join(OUT_DIR, "rf_model.joblib"))
with open(os.path.join(OUT_DIR, "feature_list.json"), "w") as f:
    json.dump(feature_cols, f, indent=2)

print("Saved model and feature list to", OUT_DIR)

# Save a small sample CSV for quick manual tests
df_sample = df[["building_id","construction_year","built_up_area_sqm","sanctioned_built_up_area",
                "occupancy_certificate","fire_noc","structural_audit_result","damage_count",
                "no_of_floors","no_of_flats","no_of_people","maintenance_per_month","latitude","longitude","label_score"]].head(50)
df_sample.to_csv("sample_buildings.csv", index=False)
print("Wrote sample_buildings.csv")
