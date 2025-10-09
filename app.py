import streamlit as st
import requests
import pandas as pd
import os
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pptx import Presentation
from pptx.util import Inches
import tempfile
import plotly.io as pio

st.set_page_config(page_title="Social Media Analysis Master App", layout="wide")
st.title("Social Media Analysis Master App")

logo_path = os.path.join(os.path.dirname(__file__), "tmu_logo.png")
if os.path.exists(logo_path):
    st.image(logo_path, width=200)
else:
    st.write("TMU Logo not found")

def save_plotly_fig(fig, filename):
    pio.write_image(fig, filename)

def create_pptx_report(df, logo_path, output_path):
    prs = Presentation()
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    slide.shapes.title.text = "TMU Social Media Analysis Report"
    slide.placeholders[1].text = "Generated October 2025"
    if os.path.exists(logo_path):
        slide.shapes.add_picture(logo_path, Inches(0.5), Inches(0.5), width=Inches(1.5))
    with tempfile.TemporaryDirectory() as tmpdirname:
        fig1 = px.line(df, x='timestamp', y='engagement_rate', title='Engagement Rate Over Time')
        path1 = os.path.join(tmpdirname, "engagement_rate.png")
        save_plotly_fig(fig1, path1)
        metrics = ['likes', 'comments', 'shares']
        sums = df[metrics].sum().reset_index()
        sums.columns = ['Metric', 'Count']
        fig2 = px.bar(sums, x='Metric', y='Count', title='Total Likes, Comments & Shares')
        path2 = os.path.join(tmpdirname, "likes_comments_shares.png")
        save_plotly_fig(fig2, path2)
        for title, img_path in {"Engagement Rate Over Time": path1, "Likes, Comments & Shares": path2}.items():
            slide_layout = prs.slide_layouts[5]
            slide = prs.slides.add_slide(slide_layout)
            slide.shapes.title.text = title
            slide.shapes.add_picture(img_path, Inches(1), Inches(1.5), width=Inches(7), height=Inches(4))
        prs.save(output_path)

tabs = st.tabs([
    "Upload", "Preview", "Dashboard", "Audience",
    "Content Insights", "Best Time", "ML Explainability",
    "Forecast", "Clustering", "Export"
])

with tabs[0]:
    st.header("CSV Upload")
    f = st.file_uploader("Upload CSV", type=["csv"])
    if f:
        resp = requests.post("http://backend:8000/upload/", files={'file': f})
        data = resp.json()
        if "rows" in data:
            st.success(f"Uploaded {data['rows']} rows!")
        else:
            st.error(data.get("error", "Upload failed."))

with tabs[1]:
    st.header("Data Preview")
    if st.button("Load Preview"):
        preview = requests.get("http://backend:8000/preview/")
        df = pd.DataFrame(preview.json())
        st.dataframe(df)

# ... (existing dashboard, audience, content, best time tabs same as before) ...

with tabs[6]:  # ML Explainability
    st.header("ML Explainability (SHAP Summary Plot)")
    if st.button("Load SHAP Plot"):
        shap_url = "http://backend:8000/ml/explain/"
        st.image(shap_url)

with tabs[7]:  # Forecast
    st.header("Engagement Forecast (Next 30 Days)")
    response = requests.get("http://backend:8000/ml/forecast/")
    forecast_data = response.json() if response.status_code == 200 else []
    if forecast_data:
        df_forecast = pd.DataFrame(forecast_data)
        df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])
        fig = px.line(df_forecast, x='ds', y='yhat', title='Forecasted Engagement')
        fig.add_scatter(x=df_forecast['ds'], y=df_forecast['yhat_upper'], mode='lines', line=dict(dash='dash'), name='Upper Bound')
        fig.add_scatter(x=df_forecast['ds'], y=df_forecast['yhat_lower'], mode='lines', line=dict(dash='dash'), name='Lower Bound')
        st.plotly_chart(fig)

with tabs[8]:  # Clustering
    st.header("Caption Clustering & Suggestions")
    response = requests.get("http://backend:8000/ml/cluster/")
    clusters = response.json() if response.status_code == 200 else {}
    for cluster_name, samples in clusters.items():
        st.subheader(cluster_name)
        for s in samples:
            st.write("-", s)

with tabs[9]:  # Export
    st.header("Export Data & Professional Report")
    preview = requests.get("http://backend:8000/preview/")
    df = pd.DataFrame(preview.json())
    if len(df) > 0:
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, "social_media_analysis.csv", "text/csv")

        if st.button("Generate and Download PPTX Report"):
            tmp_report_path = "tmu_social_media_report.pptx"
            create_pptx_report(df, logo_path, tmp_report_path)
            with open(tmp_report_path, "rb") as f:
                st.download_button("Download TMU PPTX Report", f, "tmu_report.pptx")
