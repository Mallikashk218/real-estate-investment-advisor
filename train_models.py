"""
India Housing - train the two models used by the Streamlit app.
Run once:   python train_models.py
Needs:      cleaned_housing_data.csv (from the cleaning script)
Creates:    housing_models.joblib
Install:    pip install pandas numpy scikit-learn joblib
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SAMPLE_ROWS = 100_000      # use fewer rows to keep training fast and the model file small
RANDOM_STATE = 42

df = pd.read_csv("cleaned_housing_data.csv")
if len(df) > SAMPLE_ROWS:
    df = df.sample(SAMPLE_ROWS, random_state=RANDOM_STATE)

# ------------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------------
CAT = ["City", "Property_Type", "Availability_Status", "Owner_Type", "Facing"]
NUM_REG = ["BHK", "Size_in_SqFt", "Floor_No", "Total_Floors", "Age_of_Property",
           "Nearby_Schools", "Nearby_Hospitals", "Transport_Score",
           "Parking_Space_Code", "Security_Code", "Amenity_Count", "Furnished_Code"]
# The classifier also sees the asking price (as price per sqft and vs the city median)
NUM_CLS = NUM_REG + ["Price_per_SqFt", "Price_vs_City_Median"]

# typical price per sqft in each city (the app needs this to work out Price_vs_City_Median)
city_median_pps = df.groupby("City")["Price_per_SqFt"].median().to_dict()


def make_pipeline(model, numeric):
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
        remainder="passthrough")
    return Pipeline([("prep", prep), ("model", model)]), CAT + numeric


def importance_by_feature(pipe):
    """Add up the one-hot columns so City etc. show as ONE feature."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pipe.named_steps["model"].feature_importances_
    out = {}
    for name, value in zip(names, imp):
        if name.startswith("cat__"):
            original = next(c for c in CAT if name.startswith(f"cat__{c}_"))
        else:
            original = name.replace("remainder__", "")
        out[original] = out.get(original, 0) + value
    return pd.Series(out).sort_values(ascending=False)


# ------------------------------------------------------------------
# 1. CLASSIFIER: Is this a Good Investment?
# ------------------------------------------------------------------
clf_pipe, clf_cols = make_pipeline(
    RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=50,
                           n_jobs=-1, random_state=RANDOM_STATE), NUM_CLS)
X_tr, X_te, y_tr, y_te = train_test_split(df[clf_cols], df["Good_Investment"],
                                          test_size=0.2, random_state=RANDOM_STATE,
                                          stratify=df["Good_Investment"])
clf_pipe.fit(X_tr, y_tr)
pred = clf_pipe.predict(X_te)
proba = clf_pipe.predict_proba(X_te)[:, 1]
clf_metrics = {"accuracy": accuracy_score(y_te, pred),
               "f1": f1_score(y_te, pred),
               "roc_auc": roc_auc_score(y_te, proba)}
print("Classifier:", {k: round(v, 3) for k, v in clf_metrics.items()})

# ------------------------------------------------------------------
# 2. REGRESSOR: estimated price (in lakhs) today
# ------------------------------------------------------------------
reg_pipe, reg_cols = make_pipeline(
    RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=100,
                          n_jobs=-1, random_state=RANDOM_STATE), NUM_REG)
X_tr, X_te, y_tr, y_te = train_test_split(df[reg_cols], df["Price_in_Lakhs"],
                                          test_size=0.2, random_state=RANDOM_STATE)
reg_pipe.fit(X_tr, y_tr)
pred = reg_pipe.predict(X_te)
reg_metrics = {"r2": r2_score(y_te, pred),
               "mae": mean_absolute_error(y_te, pred),
               # error you would get by always guessing the average price
               "baseline_mae": mean_absolute_error(y_te, np.full(len(y_te), y_tr.mean()))}
print("Regressor:", {k: round(v, 3) for k, v in reg_metrics.items()})

# ------------------------------------------------------------------
# SAVE EVERYTHING THE APP NEEDS
# ------------------------------------------------------------------
joblib.dump({"clf": clf_pipe, "reg": reg_pipe,
             "clf_cols": clf_cols, "reg_cols": reg_cols,
             "city_median_pps": city_median_pps,
             "clf_metrics": clf_metrics, "reg_metrics": reg_metrics,
             "clf_importance": importance_by_feature(clf_pipe),
             "reg_importance": importance_by_feature(reg_pipe),
             "price_range": (float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max()))},
            "housing_models.joblib", compress=3)
print("Saved housing_models.joblib")
