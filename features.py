import pandas as pd
import json
from sklearn.cluster import KMeans
import numpy as np


def add_features(df):
    """Original feature engineering function"""
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['caption_length'] = df['caption_text'].fillna('').astype(str).str.len()

    # Parse hashtags and count them
    def count_hashtags(hashtag_str):
        try:
            hashtags = json.loads(hashtag_str) if isinstance(hashtag_str, str) else []
            return len(hashtags) if isinstance(hashtags, list) else 0
        except:
            return 0

    df['hashtag_count'] = df['hashtags'].apply(count_hashtags)

    # Calculate engagement rate
    df['total_engagement'] = df['likes'].fillna(0) + df['comments'].fillna(0) + df['shares'].fillna(0)
    df['engagement_rate'] = df['total_engagement'] / df['impressions'].fillna(1)
    df['engagement_rate'] = df['engagement_rate'].fillna(0)

    return df


def add_advanced_features(df):
    """Advanced social media analysis features"""
    # Ensure basic features are present
    df = add_features(df)

    # Advanced engagement metrics
    df['ctr'] = (df['total_engagement'] / df['reach'].fillna(1)).fillna(0)  # Click-through rate proxy
    df['viral_coefficient'] = (df['shares'] / df['impressions'].fillna(1)).fillna(0)
    df['comment_rate'] = (df['comments'] / df['impressions'].fillna(1)).fillna(0)

    # Content performance scoring
    df['performance_score'] = (
            df['engagement_rate'] * 0.4 +
            df['viral_coefficient'] * 0.3 +
            df['comment_rate'] * 0.3
    )

    # Time-based features
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_prime_time'] = df['hour'].isin([18, 19, 20, 21]).astype(int)  # 6-9 PM

    # Content categorization
    def categorize_post_length(length):
        if length < 50:
            return 'Short'
        elif length < 150:
            return 'Medium'
        else:
            return 'Long'

    df['content_length_category'] = df['caption_length'].apply(categorize_post_length)

    # Audience segmentation using clustering
    features_for_clustering = df[['follower_count', 'engagement_rate', 'hashtag_count']].fillna(0)

    if len(df) >= 3:  # Need at least 3 samples for clustering
        kmeans = KMeans(n_clusters=min(3, len(df)), random_state=42)
        df['audience_segment'] = kmeans.fit_predict(features_for_clustering)
    else:
        df['audience_segment'] = 0

    # Post type performance
    post_type_performance = df.groupby('post_type')['engagement_rate'].mean().to_dict()
    df['post_type_avg_performance'] = df['post_type'].map(post_type_performance)

    return df
