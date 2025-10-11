import shap
import pickle
import pandas as pd

def explain_model(df, model_path="engagement_model.pkl", num_samples=100):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X = df[['hour', 'dayofweek', 'caption_length', 'hashtag_count']].iloc[:num_samples]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return shap_values, X
