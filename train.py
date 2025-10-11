import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pickle

def train_model(df):
    features = ['hour', 'dayofweek', 'caption_length', 'hashtag_count']
    X = df[features]
    y = df['engagement_rate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)
    with open("engagement_model.pkl", "wb") as f:
        pickle.dump(model, f)
