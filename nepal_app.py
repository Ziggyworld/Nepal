import streamlit as st
import pickle
import io
import numpy as np
import pandas as pd

st.title("Building Damage Prediction App")

def main_page():
    # Load the saved model
    with open("nepal_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)

    model = artifacts["model"]
    ohe = artifacts["ohe"]
    cat_cols = artifacts["cat_cols"]
    numeric_cols = artifacts["numeric_cols"]
    feature_columns = artifacts["feature_columns"]
    THRESHOLD = artifacts.get("threshold")

    # Input fields for the features
    st.header("Input Features")
    st.subheader('Please note: The Model can accurately predict 70% of the damage grade of buildings in Nepal only.')

    age_building = st.number_input("Age of Building (0-999)", min_value=0, max_value=999)
    foundation_type = st.selectbox("Foundation Type", options=['Mud mortar-Stone/Brick', 'Cement-Stone/Brick', 'RC', 'Other', 'Bamboo/Timber'])
    ground_floor_type = st.selectbox("Ground Floor Type", options=['Mud', 'Brick/Stone', 'RC', 'Timber', 'Other'])
    height_ft_pre_eq = st.number_input("Height (ft) Pre-Earthquake (6-99)", min_value=6, max_value=99)
    land_surface_condition = st.selectbox("Land Surface Condition", options=['Flat', 'Moderate slope', 'Steep slope'])
    other_floor_type = st.selectbox("Other Floor Type", options=['Timber/Bamboo-Mud', 'Timber-Planck', 'RCC/RB/RBC', 'Not applicable'])
    plan_configuration = st.selectbox("Plan Configuration", options=['Rectangular', 'Square', 'L-shape', 'Multi-projected', 'Others', 'U-shape', 'T-shape', 'H-shape', 'E-shape', 'Building with Central Courtyard'])
    plinth_area_sq_ft = st.number_input("Plinth Area (sq ft) (70-4995)", min_value=70, max_value=4995)
    position = st.selectbox("Position", options=['Not attached', 'Attached-1 side', 'Attached-2 side', 'Attached-3 side'])
    roof_type = st.selectbox("Roof Type", options=['Bamboo/Timber-Heavy roof', 'Bamboo/Timber-Light roof', 'RCC/RB/RBC'])
    superstructure = st.selectbox("Superstructure", options=['mud_mortar_stone', 'cement_mortar_brick', 'rc_non_engineered', 'stone_flag', 'adobe_mud', 'mud_mortar_brick', 'timber', 'cement_mortar_stone', 'rc_engineered', 'bamboo', 'other'])

    def preprocess_input(input_df):
        # apply same OHE + numeric concat + column alignment as training
        ohe_arr = ohe.transform(input_df[cat_cols])
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(cat_cols), index=input_df.index)
        num_df = input_df[numeric_cols].astype(float)
        X = pd.concat([num_df, ohe_df], axis=1)
        # ensure all training columns present in same order
        for c in feature_columns:
            if c not in X.columns:
                X[c] = 0
        X = X[feature_columns]
        return X

    # Create a dictionary of the input data
    input_data = {
        'age_building': age_building,
        'foundation_type': foundation_type,
        'ground_floor_type': ground_floor_type,
        'height_ft_pre_eq': height_ft_pre_eq,
        'land_surface_condition': land_surface_condition,
        'other_floor_type': other_floor_type,
        'plan_configuration': plan_configuration,
        'plinth_area_sq_ft': plinth_area_sq_ft,
        'position': position,
        'roof_type': roof_type,
        'superstructure': superstructure
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    if st.button("Predict"):
        X_in = preprocess_input(input_df)
        y_proba = model.predict_proba(X_in)[:, 1]
        for t in [THRESHOLD]:
            y_pred_adj = (y_proba > t).astype(int)
        if y_pred_adj[0] == 1:
            st.info("Prediction: was affected by earthquake")
        else:
            st.info("Prediction: was not affected by earthquake")


def bulk_upload_page():

    st.header("Bulk CSV Upload — Earthquake Damage Prediction")

    @st.cache_data(ttl=3600)
    def load_artifacts(path="nepal_artifacts.pkl"):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Cannot load artifacts: {e}")
            return None

    artifacts = load_artifacts()
    if artifacts is None:
        st.stop()

    model = artifacts["model"]
    ohe = artifacts["ohe"]
    cat_cols = artifacts.get("cat_cols", [])
    numeric_cols = artifacts.get("numeric_cols", [])
    feature_columns = artifacts.get("feature_columns", None)
    THRESHOLD = 0.4 #artifacts.get("threshold")

    st.markdown("Upload a CSV with the same feature columns used during training.")
    uploaded = st.file_uploader("Choose CSV file", type="csv", accept_multiple_files=False)
    threshold = st.slider("Prediction threshold (probability -> positive)", 0.3, 0.4, 0.5, 0.6)

    def preprocess_input(df: pd.DataFrame):
        missing = [c for c in (numeric_cols + cat_cols) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df_num = df[numeric_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0)
        df_cat = df[cat_cols].astype(str).fillna("missing")

        ohe_arr = ohe.transform(df_cat)
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(cat_cols), index=df.index)

        X = pd.concat([df_num, ohe_df], axis=1)

        if feature_columns is not None:
            for c in feature_columns:
                if c not in X.columns:
                    X[c] = 0
            X = X[feature_columns]
        return X

    if uploaded is not None:
        try:
            df = pd.read_csv(io.BytesIO(uploaded.read()))
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return

        st.write("Uploaded file preview")
        st.dataframe(df.head())

        if st.button("Run bulk predictions"):
            try:
                X_in = preprocess_input(df)
            except Exception as e:
                st.error(f"Preprocessing error: {e}")
                return

            proba = model.predict_proba(X_in)[:, 1]
            #for t in [THRESHOLD]:
            pred = (proba >THRESHOLD).astype(int)

            out = df.copy()
            out["severe_proba"] = np.round(proba, 4)
            out["severe_pred"] = pred

            st.success("Predictions complete")
            st.write("Prediction counts")
            st.write(out["severe_pred"].value_counts())

            st.dataframe(out.head(50))

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv_bytes, file_name="bulk_predictions.csv", mime="text/csv")
    else:
        st.info("Waiting for CSV upload.")

try:
    pages = {
        "Main": main_page,            
        "Bulk Upload": bulk_upload_page
    }
    choice = st.sidebar.selectbox("Page", list(pages.keys()))
    pages[choice]()
except NameError:
    pass