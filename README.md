🚰 Predictive Contamination Risk Mapping
⚠️ Forecast Unsafe Water Risks + Cluster Sites Based on Water Quality (TDS & pH)
🧠 Project Overview
This project combines time-series modeling and unsupervised clustering to:

Predict unsafe water risk for the next 3 days using sensor and weather data.

Detect patterns and anomalies using unsupervised clustering.

Provide site-level and master-level predictions with an API interface.

Offer visual risk insights using clustering plots and summary statistics.

🎯 Core Objectives
Predict future unsafe days (next 3 days) for each site.

Cluster sensor data (TDS & pH) across sites under a master region.

Provide interactive API-based access to predictions and clustering.

⚙️ Tech Stack
Task	Tools Used
ML Models	LSTM, XGBoost, KMeans, DBSCAN
API Framework	FastAPI
Data Storage	MongoDB
Data Processing	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Environment Management	Conda

🔗 API Modules
🔹 1. Prediction APIs (LSTM + XGBoost)
Endpoint	Description
/train_and_predict	Train and predict for a single site
/predict_only	Predict next 3 unsafe/safe days using saved model
/retrain_all	Retrain models for all sites
/train_and_predict_master	Train model using aggregated data from child sites
/predict_only_master	Predict future unsafe days for a master site
/retrain_all_masters	Retrain all master-site models

📘 Each model predicts 3-day unsafe/safe labels (0 = safe, 1 = unsafe) using:

LSTM forecast of TDS & pH

XGBoost classifier trained on predicted sequences

🔹 2. Clustering API (Master Site-Based Analysis)
Endpoint	Description
/cluster/{master_site}	Cluster water quality (TDS & pH) from all child sites under a master site

📤 Response:

json
Copy
Edit
{
  "optimal_k": 3,
  "elbow_plot_base64": "<image_base64>",
  "cluster_plot_base64": "<image_base64>",
  "cluster_summary": [
    {
      "cluster": 0,
      "tds_mean": 86.12,
      "tds_min": 56.0,
      "tds_max": 110.0,
      "tds_std": 10.45,
      "ph_mean": 7.21,
      "ph_min": 6.9,
      "ph_max": 7.5,
      "ph_std": 0.14
    }
  ]
}
📊 Cluster Insights Returned:

Optimal K using Elbow Method

WCSS Plot (K vs Inertia)

Clustering plot (TDS vs pH)

Summary stats (mean, min, max, std) for each cluster

🛠️ Internals:

Uses StandardScaler

Auto-selects best K using WCSS Elbow Method

Plots converted to base64 PNGs

Uses master_sites.childSites to fetch related sites

📦 Folder Structure
bash
.
├── fastapi_app.py               # Prediction APIs
├── clustering_api.py            # Clustering API logic
├── model_utils.py               # LSTM + XGB training/inference functions
├── requirements.txt
├── models/                      # Trained model files (.h5, .pkl)
├── utils/                       # Common helpers (e.g., image encoding)
└── README.md
🔧 Setup Instructions
▶️ Create Environment
bash
conda create -n water-risk-env python=3.10
conda activate water-risk-env
pip install -r requirements.txt
▶️ Start FastAPI App
bash
uvicorn fastapi_app:app --reload
For Clustering APIs:
bash
uvicorn clustering_api:app --reload
📬 Open Swagger UI:
http://127.0.0.1:8000/docs

MONGODB Remote Access
from pymongo import MongoClient
client = MongoClient("mongodb://<your_ip>:27017")


🧩 Future Enhancements
Add rainfall, temperature data to LSTM model.

Create interactive map overlay for risk visualizations.

Auto-schedule model training & prediction via cron.

Integrate alert system (e.g., SMS or email notifications).

🧑‍💻 Author
Shreyansh Srivastava
ML Developer • Startup Enthusiast • Innovative Thinker
