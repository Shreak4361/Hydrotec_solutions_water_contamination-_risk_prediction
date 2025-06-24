from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
from io import BytesIO
import base64

from tensorboard import summary

app = FastAPI()

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client.arosia_db

def encode_plot_as_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

@app.get("/cluster/{master_site}")
def cluster_by_master_site(master_site: str):
    # Load master site mappings
    master_sites_cursor = db.master_sites.find()
    site_docs = {str(site['_id']): site.get('name') for site in db.sites.find()}
    site_to_master = {}

    for master in master_sites_cursor:
        for child in master["childSites"]:
            site_to_master[str(child.id)] = {
                "masterSite": master["name"],
                "masterId": master["_id"],
                "siteName": site_docs.get(str(child.id))
            }

    # Get child site ids for given master site
    child_ids = [sid for sid, val in site_to_master.items() if val["masterSite"] == master_site]

    if not child_ids:
        raise HTTPException(status_code=404, detail="No data found for given master site.")

    # Fetch DAILY records for those sites
    cursor = db.dispenser_records.find({
        "recordType": "DAILY",
        "siteId": {"$in": child_ids}
    })

    rows = []
    for rec in cursor:
        sid = rec.get("siteId")
        info = site_to_master.get(sid, {})
        rows.append({
            "tds": str(rec.get("tds", "")).strip(),
            "ph": str(rec.get("ph", "")).strip(),
            "site": rec.get("site"),
            "date": rec.get("date"),
            "masterSite": info.get("masterSite"),
            "siteName": info.get("siteName"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise HTTPException(status_code=404, detail="No records found for clustering.")

    # Clean and preprocess
    df['tds'].replace('', pd.NA, inplace=True)
    df['ph'].replace('', pd.NA, inplace=True)
    df['tds'] = pd.to_numeric(df['tds'], errors='coerce')
    df['ph'] = pd.to_numeric(df['ph'], errors='coerce')
    df.dropna(subset=['tds', 'ph'], inplace=True)

    if df.empty:
        raise HTTPException(status_code=422, detail="All TDS/pH records are NaN.")

    # Clustering
    features = ['tds', 'ph']
    X = df[features]
    X_scaled = StandardScaler().fit_transform(X)

    # WCSS and Elbow
    wcss = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)

    kl = KneeLocator(range(2, 11), wcss, curve="convex", direction="decreasing")
    optimal_k = int(kl.elbow or 3)

    # Plot Elbow
    fig1 = plt.figure()
    plt.plot(range(2, 11), wcss, 'bo--')
    plt.title("Elbow Method")
    plt.xlabel("K")
    plt.ylabel("WCSS")
    elbow_plot_b64 = encode_plot_as_base64(fig1)
    plt.close(fig1)

    # Final KMeans
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Plot clusters
    fig2 = plt.figure()
    plt.scatter(df['tds'], df['ph'], c=df['cluster'], cmap='tab10', s=50, alpha=0.7)
    plt.title("TDS vs pH Clusters")
    plt.xlabel("TDS")
    plt.ylabel("pH")
    cluster_plot_b64 = encode_plot_as_base64(fig2)
    plt.close(fig2)

    # Cluster Summary
    summary_df = df.groupby("cluster")[["tds", "ph"]].agg(['mean', 'min', 'max', 'std']).round(2)
    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
    summary_df = summary_df.reset_index()
    cluster_summary = summary_df.to_dict(orient="records")

    return {
        "master_site": master_site,
        "child_sites_count": len(child_ids),
        "records_used": len(df),
        "optimal_k": optimal_k,
        "elbow_plot_base64": elbow_plot_b64,
        "cluster_plot_base64": cluster_plot_b64,
        "cluster_summary": cluster_summary
    }