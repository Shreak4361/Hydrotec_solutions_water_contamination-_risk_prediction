# fastapi_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from model_utils import fetch_latest_data_for_master_site, prepare_data_for_master_site, prepare_data_for_site, train_models_for_master_site, train_models_for_site, fetch_latest_data_for_site

app = FastAPI()

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

class SiteRequest(BaseModel):
    site_name: str

@app.post("/train_and_predict")
def train_and_predict(req: SiteRequest):
    site_name = req.site_name

    # Step 1: Fetch and prepare data
    try:
        X_seq, Y_seq, Y_unsafe = prepare_data_for_site(site_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data preparation failed: {e}")

    # Step 2: Train models
    try:
        lstm_model, xgb_model = train_models_for_site(site_name, X_seq, Y_seq, Y_unsafe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {e}")

    return {"message": f"Training complete for {site_name}"}

@app.post("/predict_only")
def predict_only(req: SiteRequest):
    site_name = req.site_name

    # Load models
    lstm_path = os.path.join(MODEL_DIR, f"lstm_{site_name}.h5")
    xgb_path = os.path.join(MODEL_DIR, f"xgb_{site_name}.pkl")

    if not os.path.exists(lstm_path) or not os.path.exists(xgb_path):
        raise HTTPException(status_code=404, detail="Models not found. Please train first.")

    # FIX: Load LSTM model without compiling to avoid 'mse' deserialization error
    try:
        lstm_model = load_model(lstm_path, compile=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load LSTM model: {e}")

    try:
        xgb_model = joblib.load(xgb_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load XGBoost model: {e}")

    # Fetch latest data
    try:
        X_latest = fetch_latest_data_for_site(site_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch latest data: {e}")

    # Predict future values
    try:
        Y_pred = lstm_model.predict(X_latest).reshape(-1, 6)
        Y_class = xgb_model.predict(Y_pred)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    result = {
        "unsafe_day+1": int(Y_class[0, 0]),
        "unsafe_day+2": int(Y_class[0, 1]),
        "unsafe_day+3": int(Y_class[0, 2])
    }
    return result

@app.post("/retrain_all")
def retrain_all():
    from model_utils import get_all_sites  # utility to return list of site names
    sites = get_all_sites()
    success, failed = [], []

    for site in sites:
        try:
            X_seq, Y_seq, Y_unsafe = prepare_data_for_site(site)
            train_models_for_site(site, X_seq, Y_seq, Y_unsafe)
            success.append(site)
        except Exception:
            failed.append(site)

    return {"trained": success, "failed": failed}


@app.post("/train_and_predict_master")
def train_and_predict_master(req: SiteRequest):
    site_name = req.site_name

    try:
        X_seq, Y_seq, Y_unsafe = prepare_data_for_master_site(site_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data preparation failed: {e}")

    try:
        lstm_model, xgb_model = train_models_for_master_site(site_name, X_seq, Y_seq, Y_unsafe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {e}")

    return {"message": f"Training complete for master site {site_name}"}


@app.post("/predict_only_master")
def predict_only_master(req: SiteRequest):
    site_name = req.site_name

    lstm_path = os.path.join(MODEL_DIR, f"lstm_{site_name}_master.h5")
    xgb_path = os.path.join(MODEL_DIR, f"xgb_{site_name}_master.pkl")

    if not os.path.exists(lstm_path) or not os.path.exists(xgb_path):
        raise HTTPException(status_code=404, detail="Models not found. Please train first.")

    try:
        lstm_model = load_model(lstm_path, compile=False)
        xgb_model = joblib.load(xgb_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

    try:
        X_latest = fetch_latest_data_for_master_site(site_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch latest data: {e}")

    try:
        Y_pred = lstm_model.predict(X_latest).reshape(-1, 6)
        Y_class = xgb_model.predict(Y_pred)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    result = {
        "unsafe_day+1": int(Y_class[0, 0]),
        "unsafe_day+2": int(Y_class[0, 1]),
        "unsafe_day+3": int(Y_class[0, 2])
    }
    return result

@app.post("/retrain_all_masters")
def retrain_all_masters():
    from model_utils import (
        get_all_master_sites,
        prepare_data_for_master_site,
        train_models_for_master_site,
    )

    master_sites = get_all_master_sites()
    success, failed = [], []

    for site in master_sites:
        try:
            X_seq, Y_seq, Y_unsafe = prepare_data_for_master_site(site)
            train_models_for_master_site(site, X_seq, Y_seq, Y_unsafe)
            success.append(site)
        except Exception as e:
            print(f"Failed to retrain master site {site}: {e}")
            failed.append(site)

    return {"trained_master_sites": success, "failed_master_sites": failed}

