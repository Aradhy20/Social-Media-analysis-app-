from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import duckdb
import pandas as pd
import os

from ml import explainer, forecast, clustering

app = FastAPI()

app.mount("/static", StaticFiles(directory="backend/static"), name="static")

TABLE = "posts"

@app.post("/upload/")
async def upload_csv(file: bytes = None):
    import io
    df = pd.read_csv(io.BytesIO(file))
    schema = ["timestamp", "platform", "post_id", "post_type", "caption_text", "hashtags", "likes", "comments",
              "shares", "saves", "impressions", "reach", "follower_count", "audience_gender_counts",
              "audience_age_buckets", "location"]
    missing = [c for c in schema if c not in df.columns]
    if missing:
        return JSONResponse({"error": f"Missing columns: {missing}"}, status_code=400)
    from etl import features
    df = features.add_features(df)
    con = duckdb.connect('db/db.duckdb')
    con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} AS SELECT * FROM df LIMIT 0")
    con.execute(f"INSERT INTO {TABLE} SELECT * FROM df")
    con.close()
    return {"rows": len(df)}

@app.get("/preview/")
def preview():
    con = duckdb.connect('db/db.duckdb')
    df = con.execute(f"SELECT * FROM {TABLE} LIMIT 100").fetchdf()
    con.close()
    return df.to_dict(orient="records")

@app.get("/ml/explain/")
def get_shap_plot():
    con = duckdb.connect('db/db.duckdb')
    df = con.execute(f"SELECT * FROM {TABLE}").fetchdf()
    con.close()
    output_path = explainer.explain_model(df)
    return FileResponse(output_path, media_type='image/png')

@app.get("/ml/forecast/")
def get_forecast():
    con = duckdb.connect('db/db.duckdb')
    df = con.execute(f"SELECT * FROM {TABLE}").fetchdf()
    con.close()
    forecast_df = forecast.forecast_engagement(df)
    return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).to_dict(orient="records")

@app.get("/ml/cluster/")
def get_clusters():
    con = duckdb.connect('db/db.duckdb')
    df = con.execute(f"SELECT * FROM {TABLE}").fetchdf()
    con.close()
    df, model, vectorizer = clustering.cluster_captions(df)
    clustered_texts = {}
    for cluster_num in range(model.n_clusters):
        samples = df[df['caption_cluster'] == cluster_num]['caption_text'].head(5).tolist()
        clustered_texts[f"Cluster {cluster_num}"] = samples
    return clustered_texts
