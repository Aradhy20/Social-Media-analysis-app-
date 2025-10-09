from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import json
import duckdb
from etl.features import add_features

app = FastAPI()

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    # Mapping based on your TMU dataset columns
    column_map = {
        'Post ID': 'post_id',
        'Publish time': 'timestamp',
        'Caption type': 'post_type',
        'Title': 'caption_text',
        'Hashtags': 'hashtags',
        'Reactions': 'likes',
        'Comments': 'comments',
        'Shares': 'shares',
        # Add or remove mappings to match your dataset
    }

    rename_dict = {k: v for k, v in column_map.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    # Required columns
    required_cols = [
        'timestamp', 'platform', 'post_id', 'post_type', 'caption_text', 'hashtags',
        'likes', 'comments', 'shares', 'saves', 'impressions', 'reach',
        'follower_count', 'audience_gender_counts', 'audience_age_buckets', 'location'
    ]

    # Add missing columns with None/defaults
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Assign platform default if missing or empty
    if 'platform' not in df.columns or df['platform'].isnull().all():
        df['platform'] = 'Facebook'  # or adapt logic

    # Format timestamps consistently
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

    # Convert hashtags to JSON list string
    def to_list_str(x):
        if pd.isna(x) or x == '':
            return '[]'
        if isinstance(x, str) and (',' in x or ' ' in x):
            tags = [tag.strip().lstrip('#') for tag in x.split(',')]
            return json.dumps([t for t in tags if t])
        return json.dumps([x])
    df['hashtags'] = df['hashtags'].apply(to_list_str)

    # Fill audience fields with empty json if missing
    df['audience_gender_counts'] = df.get('audience_gender_counts', '{}')
    df['audience_age_buckets'] = df.get('audience_age_buckets', '{}')
    df['location'] = df['location'].fillna('Unknown') if 'location' in df.columns else 'Unknown'

    return df[required_cols]

@app.post("/upload/")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(pd.io.common.BytesIO(contents))
    df = preprocess_df(df)
    df = add_features(df)  # feature engineering

    con = duckdb.connect('db/db.duckdb')
    con.execute("CREATE TABLE IF NOT EXISTS posts AS SELECT * FROM df LIMIT 0")
    con.execute("INSERT INTO posts SELECT * FROM df")
    con.close()

    return {"rows": len(df)}

@app.get("/preview/")
def preview():
    con = duckdb.connect('db/db.duckdb')
    df = con.execute("SELECT * FROM posts LIMIT 10").fetchdf()
    con.close()
    return df.to_dict(orient="records")
