"""
rules_engine.py
---------------
Applies NMMC Building Compliance rules to a building record.
Returns compliance score (0–100) and flags describing penalties.
"""

from datetime import datetime

def apply_nmmc_rules(row: dict, current_year: int = None):
    if current_year is None:
        current_year = datetime.now().year

    score = 100.0
    flags = []

    # --- Age & Structural Safety
    age = current_year - int(row.get("construction_year", current_year))
    if age > 50 or int(row.get("structural_audit_result", 1)) == 0:
        score -= 30
        flags.append("structural_issue_or_aged_building")
    elif age > 30:
        score -= 10
        flags.append("aged_building_minor")

    # --- Occupancy Certificate
    if int(row.get("occupancy_certificate", 1)) == 0:
        score -= 20
        flags.append("no_occupancy_certificate")

    # --- Fire NOC
    if int(row.get("fire_noc", 1)) == 0:
        score -= 15
        flags.append("no_fire_noc")

    # --- Overbuilt FSI (>2%)
    built_up = float(row.get("built_up_area_sqm", 0))
    sanctioned = float(row.get("sanctioned_built_up_area", 1))
    if (built_up - sanctioned) / sanctioned > 0.02:
        score -= 10
        flags.append("excess_FSI")

    # --- Clamp final
    score = max(0.0, score)
    return score, flags
