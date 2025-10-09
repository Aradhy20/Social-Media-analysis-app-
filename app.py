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

# TMU logo display
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
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        resp = requests.post("http://backend:8000/upload/", files={'file': uploaded_file})
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

with tabs[2]:
    st.header("Engagement Over Time")
    preview = requests.get("http://backend:8000/preview/")
    df = pd.DataFrame(preview.json())
    if len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Line Chart (Plotly): Engagement Rate Over Time")
            fig = px.line(df, x='timestamp', y='engagement_rate', title='Engagement Rate Over Time')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Bar Chart (Matplotlib): Likes, Comments, Shares")
            metrics = ['likes', 'comments', 'shares']
            sums = df[metrics].sum()
            fig2, ax = plt.subplots()
            ax.bar(metrics, sums, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_title("Total Likes, Comments, Shares")
            st.pyplot(fig2)

with tabs[3]:
    st.header("Audience Insights")
    if len(df) > 0:
        gender_data = df['audience_gender_counts'].apply(eval)
        total_male = sum(d.get('m', 0) for d in gender_data)
        total_female = sum(d.get('f', 0) for d in gender_data)
        gender_df = pd.DataFrame({'Gender': ['Male', 'Female'], 'Count': [total_male, total_female]})
        st.subheader("Audience Gender Distribution (Plotly Pie)")
        fig = px.pie(gender_df, names='Gender', values='Count')
        st.plotly_chart(fig, use_container_width=True)
        age_data = df['audience_age_buckets'].apply(eval)
        age_totals = {}
        for d in age_data:
            for age, count in d.items():
                age_totals[age] = age_totals.get(age, 0) + count
        age_df = pd.DataFrame({'Age Group': list(age_totals.keys()), 'Count': list(age_totals.values())})
        st.subheader("Audience Age Distribution (Matplotlib Horizontal Bar)")
        fig3, ax3 = plt.subplots()
        ax3.barh(age_df['Age Group'], age_df['Count'], color='skyblue')
        ax3.set_xlabel('Count')
        ax3.set_title('Audience Age Distribution')
        st.pyplot(fig3)

with tabs[4]:
    st.header("Content Insights")
    if len(df) > 0:
        st.subheader("Post Type Engagement (Bar Chart - Matplotlib)")
        post_engagement = df.groupby('post_type')['engagement_rate'].mean().reset_index()
        fig4, ax4 = plt.subplots()
        sns.barplot(x='post_type', y='engagement_rate', data=post_engagement, ax=ax4)
        ax4.set_title("Average Engagement Rate by Post Type")
        st.pyplot(fig4)
        st.subheader("Hashtag Count vs Engagement Rate (Plotly Scatter)")
        fig5 = px.scatter(df, x='hashtag_count', y='engagement_rate',
                          labels={'hashtag_count': 'Hashtag Count', 'engagement_rate': 'Engagement Rate'},
                          title='Hashtag Count vs Engagement Rate')
        st.plotly_chart(fig5, use_container_width=True)
        st.subheader("Engagement Rate by Day of Week (Boxplot - Matplotlib)")
        fig6, ax6 = plt.subplots()
        sns.boxplot(x='dayofweek', y='engagement_rate', data=df, ax=ax6)
        ax6.set_title("Engagement Rate Distribution by Day of Week")
        ax6.set_xlabel("Day of Week (0=Monday)")
        ax6.set_ylabel("Engagement Rate")
        st.pyplot(fig6)

with tabs[5]:
    st.header("Best Posting Times")
    if len(df) > 0:
        best_hours = df.groupby('hour')['engagement_rate'].mean().sort_values(ascending=False).head(5)
        st.write("Top 5 Best Hours to Post (Engagement Rate)")
        st.bar_chart(best_hours)

with tabs[6]:
    st.header("ML Explainability (SHAP Summary Plot)")
    if st.button("Load SHAP Plot"):
        shap_url = "http://backend:8000/ml/explain/"
        st.image(shap_url)

with tabs[7]:
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

with tabs[8]:
    st.header("Caption Clustering & Suggestions")
    response = requests.get("http://backend:8000/ml/cluster/")
    clusters = response.json() if response.status_code == 200 else {}
    for cluster_name, samples in clusters.items():
        st.subheader(cluster_name)
        for s in samples:
            st.write("-", s)

with tabs[9]:
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
