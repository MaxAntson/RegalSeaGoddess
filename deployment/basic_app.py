# app.py
import json

import numpy as np
import streamlit as st
import xgboost as xgb


@st.cache_resource
def load_html_map():
    with open("../img/nudibranch_map.html", "r") as f:
        return f.read()


@st.cache_resource
def load_model_and_features():
    clf = xgb.XGBClassifier()
    clf.load_model("./model.json")
    with open("./feature_names.json", "r") as f:
        feature_names = json.load(f)
    return clf, feature_names


@st.cache_resource
def load_prediction_threshold():
    with open("./threshold.txt", "r") as f:
        return float(f.read().strip())


html_map = load_html_map()
PREDICTION_THRESHOLD = load_prediction_threshold()
clf, feature_names = load_model_and_features()


st.set_page_config(page_title="RSG SDM", page_icon="🐌")
st.title("The Regal Sea Goddess - Species Distribution")
st.write(
    """
This is a simple app to demonstrate the species distribution model for the regal sea goddess.
"""
)
st.header("Accessible Area - Habitat Suitability Predictions")
st.write(
    """The model predicts the suitability of habitat for the regal sea goddess based on environmental variables such as bathymetry, sea surface temperature, salinity, chlorophyll concentration, and distance to shore."""
)
st.write(
    "The map below shows the predicted suitability. The map covers the known accessible range of the regal sea goddess - from the east coast of the Americas to the Black Sea."
)
st.write("The red points are actual observations of the regal sea goddess.")
st.components.v1.html(html_map, height=400, width=1000, scrolling=False)


st.header("Predict Habitat Suitability")
st.write(
    "Try adjust the sliders to see if an environment may be suitable for the regal sea goddess!"
)
feature_display = [
    ("Bathymetry (metres below sea level)", "bathymetry"),
    ("Mean Sea Surface Temperature (°C)", "mean_sst"),
    ("Sea Surface Temperature Range (°C)", "range_sst"),
    ("Mean Salinity", "mean_salinity"),
    ("Mean Chlorophyll-a (mg/m³)", "mean_chlorophyll_a"),
    ("Mean Slope", "mean_slope"),
    ("Distance to Shore (metres)", "distance_to_shore_m"),
]
feature_ranges = {
    "bathymetry": (-1000, 0),
    "mean_sst": (4, 30),
    "range_sst": (3, 30),
    "mean_salinity": (4, 40),
    "mean_chlorophyll_a": (0, 6),
    "mean_slope": (0, 23),
    "distance_to_shore_m": (0, 100000),
}
default_values = {
    "bathymetry": -100,
    "mean_sst": 25,
    "range_sst": 15,
    "mean_salinity": 35,
    "mean_chlorophyll_a": 2,
    "mean_slope": 10,
    "distance_to_shore_m": 100,
}

user_inputs = {}
for display, key in feature_display:
    min_val, max_val = feature_ranges[key]
    user_inputs[key] = st.slider(
        label=display,
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(default_values[key]),
        step=0.1 if max_val - min_val < 100 else 1.0,
    )

if st.button("Predict"):
    input_ordered = [user_inputs[k] for k in feature_names]
    X = np.array(input_ordered).reshape(1, -1)
    pred = clf.predict_proba(X)[0, 1]
    if pred > PREDICTION_THRESHOLD:
        label = "Possibly suitable!"
    else:
        label = "Likely not suitable."
    if label == "Likely not suitable.":
        st.error(f"Predicted suitability: {label}")
    else:
        st.success(f"Predicted suitability: {label}")
