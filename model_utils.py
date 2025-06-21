# model_utils.py

import pandas as pd
import numpy as np
import os
import joblib
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Mongo setup
client = MongoClient("mongodb://206.189.137.208:27017")
db = client.arosia_db

scaler = MinMaxScaler()


def prepare_data_for_site(site_name, n_readings=5, n_targets=3):
    df = pd.DataFrame(list(db.dispenser_records.find({"recordType": "DAILY", "site": site_name})))
    if df.empty:
        raise ValueError("No data found for this site")

    df = df.sort_values("date").reset_index(drop=True)
    df['tds'] = df['tds'].astype(str).str.strip()
    df['ph'] = df['ph'].astype(str).str.strip()
    df['tds'].replace('', pd.NA, inplace=True)
    df['ph'].replace('', pd.NA, inplace=True)
    df['tds'] = pd.to_numeric(df['tds'], errors='coerce')
    df['ph'] = pd.to_numeric(df['ph'], errors='coerce')
    df.dropna(subset=['tds', 'ph'], inplace=True)

    df['unsafe'] = df.apply(lambda row: row['tds'] > 200 or row['ph'] < 6.5 or row['ph'] > 8.5, axis=1)
    if df.empty or len(df) < n_readings+ n_targets:
        raise ValueError("Not enough data for training.")

    df[['tds_scaled', 'ph_scaled']] = scaler.fit_transform(df[['tds', 'ph']])

    X_seq, Y_seq, Y_unsafe = [], [], []

    for i in range(len(df) - n_readings - n_targets + 1):
        X_chunk = df.iloc[i:i+n_readings][['tds_scaled', 'ph_scaled']].values
        Y_chunk = df.iloc[i+n_readings:i+n_readings+n_targets][['tds', 'ph']].values
        U_chunk = df.iloc[i+n_readings:i+n_readings+n_targets]['unsafe'].values

        X_seq.append(X_chunk)
        Y_seq.append(Y_chunk)
        Y_unsafe.append(U_chunk)

    return np.array(X_seq), np.array(Y_seq), np.array(Y_unsafe)


def train_models_for_site(site_name, X_seq, Y_seq, Y_unsafe):
    # Train LSTM
    model = Sequential()
    model.add(Input(shape=(X_seq.shape[1], X_seq.shape[2])))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(6))
    model.compile(optimizer=Adam(), loss='mse')
    model.fit(X_seq, Y_seq.reshape(-1, 6), epochs=20, batch_size=32, validation_split=0.2, verbose=0)

    model.save(os.path.join(MODEL_DIR, f"lstm_{site_name}.h5"))

    # Train XGBoost Classifier
    X_class = Y_seq.reshape(-1, 6)
    Y_class = pd.DataFrame(Y_unsafe, columns=['unsafe_day+1', 'unsafe_day+2', 'unsafe_day+3'])

    X_train, X_test, Y_train, Y_test = train_test_split(X_class, Y_class, test_size=0.2, random_state=42)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    clf = MultiOutputClassifier(xgb)
    clf.fit(X_train, Y_train)

    joblib.dump(clf, os.path.join(MODEL_DIR, f"xgb_{site_name}.pkl"))

    return model, clf


def fetch_latest_data_for_site(site_name, n_timesteps=5):
    df = pd.DataFrame(list(db.dispenser_records.find({"recordType": "DAILY", "site": site_name})))
    df = df.sort_values("date").reset_index(drop=True)
    df['tds'] = df['tds'].astype(str).str.strip()
    df['ph'] = df['ph'].astype(str).str.strip()
    df['tds'].replace('', pd.NA, inplace=True)
    df['ph'].replace('', pd.NA, inplace=True)
    df['tds'] = pd.to_numeric(df['tds'], errors='coerce')
    df['ph'] = pd.to_numeric(df['ph'], errors='coerce')
    df.dropna(subset=['tds', 'ph'], inplace=True)

    df[['tds_scaled', 'ph_scaled']] = scaler.transform(df[['tds', 'ph']])

    latest = df.tail(n_timesteps)[['tds_scaled', 'ph_scaled']].values
    return latest.reshape(1, n_timesteps, 2)


def get_all_sites():
    return db.dispenser_records.distinct("site", {"recordType": "DAILY"})

from bson import ObjectId
from bson.dbref import DBRef

from bson import DBRef, ObjectId
def prepare_data_for_master_site(master_site_name, n_readings=5, n_targets=3):
    master_doc = db.master_sites.find_one({"name": master_site_name})
    if not master_doc:
        raise ValueError(f"Master site '{master_site_name}' not found.")

    child_ids = []
    for ref in master_doc.get("childSites", []):
        if isinstance(ref, DBRef):
            child_ids.append(str(ref.id))  # Convert ObjectId to string
        elif isinstance(ref, dict) and "$id" in ref:
            child_ids.append(str(ref["$id"]))  # Convert ObjectId to string

    if not child_ids:
        raise ValueError("No child site IDs found for this master site.")

    print(f"[DEBUG] Found {len(child_ids)} child site IDs for master site '{master_site_name}'")

    # Fetch records using string IDs
    df = pd.DataFrame(list(db.dispenser_records.find({
        "recordType": "DAILY",
        "siteId": {"$in": child_ids}
    })))

    print(f"[DEBUG] Total records fetched: {len(df)}")

    if df.empty:
        raise ValueError("No data found for this master site")

    df = df.sort_values("date").reset_index(drop=True)

    # Clean and convert
    df['tds'] = df['tds'].astype(str).str.strip()
    df['ph'] = df['ph'].astype(str).str.strip()
    df['tds'].replace('', pd.NA, inplace=True)
    df['ph'].replace('', pd.NA, inplace=True)
    df['tds'] = pd.to_numeric(df['tds'], errors='coerce')
    df['ph'] = pd.to_numeric(df['ph'], errors='coerce')
    df.dropna(subset=['tds', 'ph'], inplace=True)

    print(f"[DEBUG] Records after cleaning: {len(df)}")

    if len(df) < n_readings + n_targets:
        raise ValueError("Not enough valid data for training.")

    df['unsafe'] = df.apply(lambda row: row['tds'] > 200 or row['ph'] < 6.5 or row['ph'] > 8.5, axis=1)
    df[['tds_scaled', 'ph_scaled']] = scaler.fit_transform(df[['tds', 'ph']])

    X_seq, Y_seq, Y_unsafe = [], [], []

    for i in range(len(df) - n_readings - n_targets + 1):
        X_chunk = df.iloc[i:i+n_readings][['tds_scaled', 'ph_scaled']].values
        Y_chunk = df.iloc[i+n_readings:i+n_readings+n_targets][['tds', 'ph']].values
        U_chunk = df.iloc[i+n_readings:i+n_readings+n_targets]['unsafe'].values

        X_seq.append(X_chunk)
        Y_seq.append(Y_chunk)
        Y_unsafe.append(U_chunk)

    return np.array(X_seq), np.array(Y_seq), np.array(Y_unsafe)

def train_models_for_master_site(master_site_name, X_seq, Y_seq, Y_unsafe):
    lstm_model = Sequential()
    lstm_model.add(Input(shape=(X_seq.shape[1], X_seq.shape[2])))
    lstm_model.add(LSTM(64, return_sequences=False))
    lstm_model.add(Dense(32, activation='relu'))
    lstm_model.add(Dense(6))
    lstm_model.compile(optimizer=Adam(), loss='mse')
    lstm_model.fit(X_seq, Y_seq.reshape(-1, 6), epochs=20, batch_size=32, validation_split=0.2, verbose=0)

    lstm_model.save(os.path.join(MODEL_DIR, f"lstm_{master_site_name}_master.h5"))

    X_class = Y_seq.reshape(-1, 6)
    Y_class = pd.DataFrame(Y_unsafe, columns=['unsafe_day+1', 'unsafe_day+2', 'unsafe_day+3'])

    X_train, X_test, Y_train, Y_test = train_test_split(X_class, Y_class, test_size=0.2, random_state=42)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    clf = MultiOutputClassifier(xgb)
    clf.fit(X_train, Y_train)

    joblib.dump(clf, os.path.join(MODEL_DIR, f"xgb_{master_site_name}_master.pkl"))

    return lstm_model, clf



def fetch_latest_data_for_master_site(master_site_name, n_timesteps=5):
    master_doc = db.master_sites.find_one({"name": master_site_name})
    if not master_doc:
        raise ValueError(f"Master site '{master_site_name}' not found.")

    # Extract child site ObjectIds and convert to strings
    child_ids = []
    for ref in master_doc.get("childSites", []):
        if isinstance(ref, DBRef):
            child_ids.append(str(ref.id))
        elif isinstance(ref, dict) and "$id" in ref:
            child_ids.append(str(ref["$id"]))

    if not child_ids:
        raise ValueError("No child site IDs found for this master site.")

    df = pd.DataFrame(list(db.dispenser_records.find({
        "recordType": "DAILY",
        "siteId": {"$in": child_ids}
    })))

    if df.empty or len(df) < n_timesteps:
        raise ValueError("Not enough data for prediction")

    df = df.sort_values("date").reset_index(drop=True)

    # Clean and convert values
    df['tds'] = df['tds'].astype(str).str.strip()
    df['ph'] = df['ph'].astype(str).str.strip()
    df['tds'].replace('', pd.NA, inplace=True)
    df['ph'].replace('', pd.NA, inplace=True)
    df['tds'] = pd.to_numeric(df['tds'], errors='coerce')
    df['ph'] = pd.to_numeric(df['ph'], errors='coerce')
    df.dropna(subset=['tds', 'ph'], inplace=True)

    if len(df) < n_timesteps:
        raise ValueError("Not enough valid records after cleaning.")

    scaler.fit(df[['tds', 'ph']])  # Fit on available clean data

    df[['tds_scaled', 'ph_scaled']] = scaler.transform(df[['tds', 'ph']])
    latest = df.tail(n_timesteps)[['tds_scaled', 'ph_scaled']].values

    return latest.reshape(1, n_timesteps, 2)

def get_all_master_sites():
    """Returns a list of all master site names"""
    return db.master_sites.distinct("name")
