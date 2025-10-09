# Social-Media-analysis-app-
# TMU Social Media Analytics Dashboard

## Overview
Interactive Streamlit dashboard for social media data analysis. Features include:
- Multi-platform KPIs
- Engagement trends & forecast
- Sentiment & topic modeling (BERTopic)
- Post clustering (t-SNE & KMeans)
- Networks (mentions & hashtags)
- Downloadable PDF & Excel reports
- Actionable recommendations

## Usage
1. Upload CSV files containing columns: 
   `date`, `post_text`, `likes`, `comments`, `shares`, `followers`, `hashtags`, `platform`, `post_type`, `user`, `mentioned_user`.
2. Apply sidebar filters: Platform, Post Type, Date range.
3. Navigate through tabs for analytics.
4. Generate PDF or Excel reports.

## Deployment
Upload folder to Streamlit Cloud. Ensure `requirements.txt` and `assets/logo.png` are included.
