from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import json
import duckdb
import os
from etl.features import add_advanced_features
from ml import explainer, forecast, clustering

app = FastAPI()

# Create static directory for serving files
os.makedirs("backend/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="backend/static"), name="static")


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess uploaded CSV to match required schema"""
    column_map = {
        'Post ID': 'post_id',
        'Publish time': 'timestamp',
        'Caption type': 'post_type',
        'Title': 'caption_text',
        'Hashtags': 'hashtags',
        'Reactions': 'likes',
        'Comments': 'comments',
        'Shares': 'shares',
        'Views': 'impressions',
        'Reach': 'reach'
    }

    # Rename columns if they exist
    rename_dict = {k: v for k, v in column_map.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    # Required columns for the app
    required_cols = [
        'timestamp', 'platform', 'post_id', 'post_type', 'caption_text', 'hashtags',
        'likes', 'comments', 'shares', 'saves', 'impressions', 'reach',
        'follower_count', 'audience_gender_counts', 'audience_age_buckets', 'location'
    ]

    # Add missing columns with defaults
    for col in required_cols:
        if col not in df.columns:
            if col == 'platform':
                df[col] = 'Facebook'
            elif col in ['hashtags', 'audience_gender_counts', 'audience_age_buckets']:
                df[col] = '[]' if col == 'hashtags' else '{}'
            elif col == 'location':
                df[col] = 'Unknown'
            else:
                df[col] = 0

    # Format timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

    # Convert hashtags to JSON list format if needed
    def format_hashtags(x):
        if pd.isna(x) or x == '':
            return '[]'
        if isinstance(x, str) and not x.startswith('['):
            tags = [tag.strip().lstrip('#') for tag in x.split(',') if tag.strip()]
            return json.dumps(tags)
        return x

    df['hashtags'] = df['hashtags'].apply(format_hashtags)

    return df[required_cols]


@app.post("/upload/")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(pd.io.common.BytesIO(contents))

    # Preprocess and add advanced features
    df = preprocess_df(df)
    df = add_advanced_features(df)

    # Store in DuckDB
    con = duckdb.connect('db/db.duckdb')
    con.execute("CREATE TABLE IF NOT EXISTS posts AS SELECT * FROM df LIMIT 0")
    con.execute("INSERT INTO posts SELECT * FROM df")
    con.close()

    return {"rows": len(df), "message": "Data uploaded and analyzed successfully"}


@app.get("/preview/")
def preview():
    con = duckdb.connect('db/db.duckdb')
    df = con.execute("SELECT * FROM posts LIMIT 100").fetchdf()
    con.close()
    return df.to_dict(orient="records")


@app.get("/analysis/overview/")
def analysis_overview():
    con = duckdb.connect('db/db.duckdb')
    df = con.execute("SELECT * FROM posts").fetchdf()
    con.close()

    if len(df) == 0:
        return {"error": "No data available"}

    overview = {
        "total_posts": len(df),
        "avg_engagement_rate": df['engagement_rate'].mean(),
        "total_likes": df['likes'].sum(),
        "total_comments": df['comments'].sum(),
        "total_shares": df['shares'].sum(),
        "avg_performance_score": df['performance_score'].mean(),
        "best_performing_post_type": df.groupby('post_type')['engagement_rate'].mean().idxmax(),
        "peak_engagement_hour": df.groupby('hour')['engagement_rate'].mean().idxmax()
    }

    return overview


@app.get("/analysis/top-hashtags/")
def top_hashtags():
    con = duckdb.connect('db/db.duckdb')
    df = con.execute("SELECT hashtags FROM posts").fetchdf()
    con.close()

    all_hashtags = []
    for hashtag_str in df['hashtags']:
        try:
            hashtags = json.loads(hashtag_str)
            all_hashtags.extend(hashtags)
        except:
            continue

    from collections import Counter
    hashtag_counts = Counter(all_hashtags)
    top_10 = dict(hashtag_counts.most_common(10))

    return top_10


@app.get("/analysis/segments/")
def audience_segments():
    con = duckdb.connect('db/db.duckdb')
    df = con.execute(
        "SELECT audience_segment, COUNT(*) as count, AVG(engagement_rate) as avg_engagement FROM posts GROUP BY audience_segment").fetchdf()
    con.close()

    return df.to_dict(orient="records")


@app.get("/analysis/performance-trends/")
def performance_trends():
    con = duckdb.connect('db/db.duckdb')
    df = con.execute(
        "SELECT DATE(timestamp) as date, AVG(engagement_rate) as avg_engagement, AVG(performance_score) as avg_performance FROM posts GROUP BY DATE(timestamp) ORDER BY date").fetchdf()
    con.close()

    return df.to_dict(orient="records")


@app.get("/analysis/content-insights/")
def content_insights():
    con = duckdb.connect('db/db.duckdb')
    df = con.execute(
        "SELECT content_length_category, AVG(engagement_rate) as avg_engagement, COUNT(*) as count FROM posts GROUP BY content_length_category").fetchdf()
    con.close()

    return df.to_dict(orient="records")


# ML endpoints (keeping your existing ones)
@app.get("/ml/explain/")
def get_shap_plot():
    con = duckdb.connect('db/db.duckdb')
    df = con.execute("SELECT * FROM posts").fetchdf()
    con.close()

    output_path = explainer.explain_model(df)
    return FileResponse(output_path, media_type='image/png')


@app.get("/ml/forecast/")
def get_forecast():
    con = duckdb.connect('db/db.duckdb')
    df = con.execute("SELECT * FROM posts").fetchdf()
    con.close()

    forecast_df = forecast.forecast_engagement(df)
    return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).to_dict(orient="records")


@app.get("/ml/cluster/")
def get_clusters():
    con = duckdb.connect('db/db.duckdb')
    df = con.execute("SELECT * FROM posts").fetchdf()
    con.close()

    df, model, vectorizer = clustering.cluster_captions(df)
    clustered_texts = {}
    for cluster_num in range(model.n_clusters):
        samples = df[df['caption_cluster'] == cluster_num]['caption_text'].head(5).tolist()
        clustered_texts[f"Cluster {cluster_num}"] = samples

    return clustered_texts
