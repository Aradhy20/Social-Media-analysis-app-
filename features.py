import pandas as pd

def add_features(df):
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Extract hour and day of week
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    # Caption length
    df['caption_length'] = df['caption_text'].str.len()
    # Hashtag count (expects list strings, e.g. '["tag1", "tag2"]')
    df['hashtag_count'] = df['hashtags'].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0)
    # Calculate engagement rate
    df['engagement_rate'] = (
        df['likes'] + df['comments'] + df['shares'] + df['saves']
    ) / df['impressions']
    return df
