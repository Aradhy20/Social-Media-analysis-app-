from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

def cluster_captions(df, num_clusters=5):
    texts = df['caption_text'].fillna("").tolist()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X = vectorizer.fit_transform(texts)
    model = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = model.fit_predict(X)
    df['caption_cluster'] = clusters
    return df, model, vectorizer
