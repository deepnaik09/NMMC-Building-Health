"""
app.py - Streamlit app for building rating & NMMC compliance

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import json
from datetime import datetime

CURRENT_YEAR = datetime.now().year

@st.cache_data
def load_artifacts():
    model = load("models/rf_model.joblib")
    with open("models/feature_list.json","r") as f:
        features = json.load(f)
    return model, features

model, model_features = load_artifacts()

st.set_page_config(page_title="NMMC Building Safety & Rating", layout="wide")
st.title("NMMC-based Building Rating & Suggestions — Navi Mumbai")

st.markdown("""
This app:
- Takes building inputs (single or CSV batch).
- Predicts a preliminary ML-based score (0-10).
- Computes deterministic NMMC-compliance penalties based on rules you specified and applies them to give final rating, compliance score, and maintenance estimate.
""")

# Sidebar input mode
mode = st.sidebar.selectbox("Input mode", ["Single input (form)", "Batch CSV upload"])

def compute_rule_penalties(row):
    """
    Applies NMMC rules as percentage penalties on a 100-point compliance baseline.
    Returns compliance_score (0-100) and list of flags/reasons.
    Rules (as provided):
     - Age: >30 years minor penalty; >50 or failed audit major penalty (30% penalty -> "structural_issue" or "aged_building")
     - No OC: 20% penalty
     - No Fire NOC: 15% penalty (applies for commercial/institutional; we apply as partial 15% if building type unknown)
     - Overbuilt > 2%: 10% penalty
     - (Parking/utilities/special zone omitted here for simplicity; can be added if data present)
    """
    score = 100.0
    flags = []

    age = CURRENT_YEAR - int(row["construction_year"])
    if age > 50 or int(row.get("structural_audit_result",1)) == 0:
        score -= 30.0
        flags.append("structural_issue_or_aged_building")
    elif age > 30:
        score -= 10.0
        flags.append("aged_building_minor")

    if int(row.get("occupancy_certificate",1)) == 0:
        score -= 20.0
        flags.append("no_oc")

    if int(row.get("fire_noc",1)) == 0:
        score -= 15.0
        flags.append("no_fire_noc")

    built_up = float(row.get("built_up_area_sqm",0))
    sanctioned = float(row.get("sanctioned_built_up_area",0)) + 1e-6
    if (built_up - sanctioned) / sanctioned > 0.02:
        score -= 10.0
        flags.append("excess_FSI")

    # clamp
    score = max(0.0, score)
    return score, flags

def engineer_for_model(df):
    df2 = df.copy()
    df2["age"] = CURRENT_YEAR - df2["construction_year"].astype(int)
    df2["overbuilt_pct"] = (df2["built_up_area_sqm"] - df2["sanctioned_built_up_area"]) / (df2["sanctioned_built_up_area"] + 1e-6)
    df2["overbuilt_flag"] = (df2["overbuilt_pct"] > 0.02).astype(int)
    df2["no_oc_flag"] = (df2["occupancy_certificate"] == 0).astype(int)
    df2["no_fire_flag"] = (df2["fire_noc"] == 0).astype(int)
    df2["struc_failed_flag"] = (df2["structural_audit_result"] == 0).astype(int)
    df2["people_per_flat"] = df2["no_of_people"] / (df2["no_of_flats"] + 1e-6)
    df2["sqm_per_floor"] = df2["built_up_area_sqm"] / (df2["no_of_floors"] + 1e-6)
    df2["damage_per_floor"] = df2["damage_count"] / (df2["no_of_floors"] + 1e-6)
    return df2

def ml_predict(df_in):
    df_features = engineer_for_model(df_in)
    X = df_features[model_features].astype(float)
    preds = model.predict(X)
    # clip 0..10
    preds = np.clip(preds, 0.0, 10.0)
    return preds

if mode == "Batch CSV upload":
    uploaded = st.file_uploader("Upload CSV with input columns (see sample)", type=["csv"])
    st.markdown("**Sample required columns**: building_id,construction_year,built_up_area_sqm,sanctioned_built_up_area,occupancy_certificate,fire_noc,structural_audit_result,damage_count,no_of_floors,no_of_flats,no_of_people,maintenance_per_month")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())
        preds = ml_predict(df)
        compliance_scores = []
        flags_list = []
        final_rating = []
        maint_est = []
        for i, row in df.iterrows():
            comp_score, flags = compute_rule_penalties(row)
            compliance_scores.append(comp_score)
            # Combine ML pred and compliance penalty:
            # Normalize compliance_score to 0-10
            comp_norm = comp_score / 10.0
            ml_pred = float(preds[i])
            # Weighted final: more weight to rule-based compliance for safety-critical domain
            final = 0.45*ml_pred + 0.55*comp_norm
            final_rating.append(round(final,2))
            flags_list.append(", ".join(flags) if flags else "ok")
            # Maintenance estimate: annualize and adjust by deficiency
            maint = row.get("maintenance_per_month", 0)
            # If compliance score < 80, suggest increase proportionally
            suggested = maint * 12 * (1 + max(0, (80 - comp_score)/100))
            maint_est.append(round(suggested,0))
        out = df.copy()
        out["ml_score_out_of_10"] = np.round(preds,2)
        out["nmmc_compliance_score_100"] = compliance_scores
        out["final_rating_out_of_10"] = final_rating
        out["maintenance_estimate_inr_ann"] = maint_est
        out["remarks"] = flags_list
        st.write(out[["building_id","ml_score_out_of_10","nmmc_compliance_score_100","final_rating_out_of_10","maintenance_estimate_inr_ann","remarks"]].head(50))
        st.download_button("Download results CSV", data=out.to_csv(index=False).encode('utf-8'), file_name="building_ratings_results.csv")
else:
    st.subheader("Enter single building details")
    with st.form("single_form"):
        b_id = st.text_input("Building ID", value="BNM10001")
        construction_year = st.number_input("Construction Year", value=1995, min_value=1900, max_value=CURRENT_YEAR)
        built_up = st.number_input("Built-up area (sqm)", value=1200.0, format="%.2f")
        sanctioned = st.number_input("Sanctioned built-up area (sqm)", value=1200.0, format="%.2f")
        occ = st.selectbox("Occupancy Certificate (1=yes,0=no)", [1,0], index=0)
        fire = st.selectbox("Fire NOC (1=yes,0=no)", [1,0], index=0)
        struc = st.selectbox("Structural Audit Result (1=pass,0=fail)", [1,0], index=1)
        damage = st.number_input("Damage count (reported issues)", value=0, min_value=0)
        floors = st.number_input("No of floors", value=6, min_value=1)
        flats = st.number_input("No of flats", value=30, min_value=1)
        people = st.number_input("No of people", value=80, min_value=1)
        maint = st.number_input("Maintenance per month (INR) for whole building", value=25000)
        submitted = st.form_submit_button("Predict")

    if submitted:
        row = {
            "building_id": b_id,
            "construction_year": int(construction_year),
            "built_up_area_sqm": float(built_up),
            "sanctioned_built_up_area": float(sanctioned),
            "occupancy_certificate": int(occ),
            "fire_noc": int(fire),
            "structural_audit_result": int(struc),
            "damage_count": int(damage),
            "no_of_floors": int(floors),
            "no_of_flats": int(flats),
            "no_of_people": int(people),
            "maintenance_per_month": float(maint),
            
        }
        df_input = pd.DataFrame([row])
        ml_pred = float(ml_predict(df_input)[0])
        compliance_score, flags = compute_rule_penalties(row)
        # Normalize compliance to 0-10
        comp_norm = compliance_score / 10.0
        final_rating = round(0.45*ml_pred + 0.55*comp_norm,2)
        # Maintenance estimate: annualize and adjust upward proportional to compliance shortfall
        suggested_ann = round(maint * 12 * (1 + max(0, (80 - compliance_score)/100)), 0)

        st.markdown("## Results")
        st.write(pd.DataFrame({
            "sr_no":[1],
            "building_id":[b_id],
            "final_rating_out_of_10":[final_rating],
            "nmmc_compliance_score_100":[compliance_score],
            "maintenance_estimate_inr_ann":[suggested_ann],
            "remarks":[", ".join(flags) if flags else "OK"]
        }))

        # Map

        st.info("ML preliminary score: {:.2f} /10. Compliance-adjusted final rating shown above.".format(ml_pred))
        st.write("Flags:", flags if flags else "OK")

st.markdown("---")
st.markdown("**Notes:** The final rating is a hybrid of an ML model trained on synthetic-but-realistic data and hard NMMC rule penalties. You can refine the weighting, add parking/utilities columns, or replace the RF with LightGBM/XGBoost for performance.")
