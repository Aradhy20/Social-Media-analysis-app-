from prophet import Prophet
import pandas as pd

def forecast_engagement(df, periods=30):
    ts_df = df.groupby('timestamp')['engagement_rate'].mean().reset_index()
    ts_df.rename(columns={'timestamp': 'ds', 'engagement_rate': 'y'}, inplace=True)

    model = Prophet()
    model.fit(ts_df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    return forecast  # DataFrame with predictions including yhat, yhat_lower, yhat_upper
