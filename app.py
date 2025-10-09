import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from bertopic import BERTopic
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import networkx as nx
from pyvis.network import Network
from prophet import Prophet
import streamlit.components.v1 as components

# -----------------------------
# TMU Colors
# -----------------------------
TMU_COLORS = ["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD","#8C564B","#E377C2","#7F7F7F"]

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="TMU Social Media Dashboard", layout="wide")

# -----------------------------
# Header
# -----------------------------
st.image("tmu_logo.png", width=150)
st.markdown("<h1 style='color:#111827'>TMU Social Media Analytics Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # -------------------------
    # Data Cleaning
    # -------------------------
    df.fillna({"likes":0,"comments":0,"shares":0,"followers":0,
               "post_text":"","hashtags":"","platform":"","user":"","mentioned_user":""}, inplace=True)
    df['post_type'] = df['post_type'].str.capitalize()
    df['date'] = pd.to_datetime(df['date'])
    df['engagement_score'] = df['likes'] + 2*df['comments'] + 3*df['shares']
    df['engagement_rate'] = df.apply(lambda x: (x['engagement_score']/x['followers']*100) if x['followers']>0 else 0, axis=1)

    # -------------------------
    # Sentiment Analysis
    # -------------------------
    df['sentiment'] = df['post_text'].apply(lambda x: "Positive" if TextBlob(x).sentiment.polarity>0.1
                                            else ("Negative" if TextBlob(x).sentiment.polarity<-0.1 else "Neutral"))

    # -------------------------
    # Topics (Unsupervised)
    # -------------------------
    topic_model = BERTopic(verbose=False)
    topics,_ = topic_model.fit_transform(df['post_text'])
    df['topic'] = topics

    # -------------------------
    # Filters
    # -------------------------
    platforms = st.multiselect("Select Platform", options=sorted(df['platform'].dropna().unique()),
                               default=sorted(df['platform'].dropna().unique()))
    post_types = st.multiselect("Select Post Type", options=sorted(df['post_type'].dropna().unique()),
                                default=sorted(df['post_type'].dropna().unique()))
    date_range = st.date_input("Select Date Range", value=(df['date'].min(), df['date'].max()),
                               min_value=df['date'].min(), max_value=df['date'].max())

    df = df[df['platform'].isin(platforms)]
    df = df[df['post_type'].isin(post_types)]
    df = df[(df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))]

    # -------------------------
    # Tabs
    # -------------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Overview / KPIs",
        "Engagement Trends",
        "Sentiment Analysis",
        "Post Clustering",
        "Engagement Forecast",
        "Topics / Unsupervised",
        "User Mentions Network",
        "Hashtag Co-Occurrence Network",
        "Recommendations & Insights"
    ])

    # -------------------------
    # Tab 1: Overview / KPIs
    # -------------------------
    with tab1:
        st.markdown("### Key Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Likes", int(df['likes'].sum()))
        col2.metric("Total Comments", int(df['comments'].sum()))
        col3.metric("Total Shares", int(df['shares'].sum()))
        col4.metric("Avg Engagement Score", round(df['engagement_score'].mean(),2))
        col5.metric("Avg Engagement Rate (%)", round(df['engagement_rate'].mean(),2))

    # -------------------------
    # Tab 2: Engagement Trends
    # -------------------------
    with tab2:
        st.markdown("## Engagement Over Time")
        engagement_fig = px.line(df.groupby('date')['engagement_score'].sum().reset_index(),
                                 x='date', y='engagement_score', title="Total Engagement Over Time",
                                 line_shape='spline', color_discrete_sequence=[TMU_COLORS[0]])
        platform_fig = px.line(df.groupby(['date','platform'])['engagement_rate'].mean().reset_index(),
                               x='date', y='engagement_rate', color='platform',
                               title="Engagement Rate by Platform", color_discrete_sequence=TMU_COLORS, markers=True)
        col1, col2 = st.columns(2)
        col1.plotly_chart(engagement_fig, use_container_width=True)
        col2.plotly_chart(platform_fig, use_container_width=True)

    # -------------------------
    # Tab 3: Sentiment Analysis
    # -------------------------
    with tab3:
        st.markdown("## Sentiment & Hashtags")
        sentiment_fig = px.pie(names=df['sentiment'].value_counts().index,
                               values=df['sentiment'].value_counts().values,
                               title="Sentiment Distribution", color_discrete_sequence=TMU_COLORS)
        text = " ".join(df['hashtags'].astype(str))
        wc = WordCloud(width=800, height=400, background_color="white", colormap='tab10').generate(text)
        fig_wc, ax = plt.subplots(figsize=(12,6))
        ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
        col1, col2 = st.columns(2)
        col1.plotly_chart(sentiment_fig, use_container_width=True)
        col2.pyplot(fig_wc)

    # -------------------------
    # Tab 4: Post Clustering
    # -------------------------
    with tab4:
        features = ['likes','comments','shares','engagement_score']
        kmeans = KMeans(n_clusters=5, random_state=42)
        df['cluster'] = kmeans.fit_predict(df[features])
        tsne_results = TSNE(n_components=2, random_state=42).fit_transform(df[features])
        df['tsne_1'], df['tsne_2'] = tsne_results[:,0], tsne_results[:,1]
        cluster_fig = px.scatter(df, x='tsne_1', y='tsne_2', color='cluster',
                                 hover_data=['post_text','engagement_score'], title="Post Clusters", color_discrete_sequence=TMU_COLORS)
        st.plotly_chart(cluster_fig, use_container_width=True)

    # -------------------------
    # Tab 5: Engagement Forecast
    # -------------------------
    with tab5:
        df_forecast = df.groupby('date')['engagement_score'].sum().reset_index()
        df_forecast.rename(columns={'date':'ds','engagement_score':'y'}, inplace=True)
        m = Prophet()
        m.fit(df_forecast)
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)
        forecast_fig = px.line(forecast, x='ds', y='yhat', title="Engagement Forecast (Next 30 Days)", color_discrete_sequence=[TMU_COLORS[1]])
        st.plotly_chart(forecast_fig, use_container_width=True)

    # -------------------------
    # Tab 6: Topics / Unsupervised
    # -------------------------
    with tab6:
        topic_counts = df['topic'].value_counts().reset_index()
        topic_counts.columns = ['Topic','Count']
        topic_fig = px.bar(topic_counts, x='Topic', y='Count', title="Topics Distribution", color='Topic', color_discrete_sequence=TMU_COLORS)
        st.plotly_chart(topic_fig, use_container_width=True)

    # -------------------------
    # Tab 7: User Mentions Network
    # -------------------------
    with tab7:
        st.markdown("## Interactive User Mentions Network")
        top_n_users = st.slider("Top N Users by Engagement", min_value=5, max_value=50, value=20, step=5)
        top_users = df[df['user'].notna()].groupby('user')['engagement_score'].sum().sort_values(ascending=False).head(top_n_users).index.tolist()
        df_network = df[df['user'].isin(top_users)]

        G = nx.DiGraph()
        for _, row in df_network.iterrows():
            source = row['user']
            targets = str(row['mentioned_user']).split(",") if row['mentioned_user'] else []
            for t in targets:
                if t.strip() in top_users:
                    G.add_edge(source.strip(), t.strip())
        net = Network(height="600px", width="100%", notebook=False, directed=True)
        net.from_nx(G)
        net.repulsion(node_distance=200, central_gravity=0.3, spring_length=200, spring_strength=0.05)
        net.save_graph("network.html")
        HtmlFile = open("network.html",'r',encoding='utf-8').read()
        components.html(HtmlFile, height=600, scrolling=True)

    # -------------------------
    # Tab 8: Hashtag Co-Occurrence Network
    # -------------------------
    with tab8:
        st.markdown("## Hashtag Co-Occurrence Network")
        all_hashtags = df['hashtags'].dropna().str.split(",", expand=True).stack().str.strip()
        top_hashtags = all_hashtags.value_counts().head(50).index.tolist()
        co_occurrence = {}
        for tags in df['hashtags'].dropna():
            tags_list = [t.strip() for t in tags.split(",") if t.strip() in top_hashtags]
            for i in range(len(tags_list)):
                for j in range(i+1, len(tags_list)):
                    pair = tuple(sorted([tags_list[i], tags_list[j]]))
                    co_occurrence[pair] = co_occurrence.get(pair,0)+1
        G = nx.Graph()
        for pair, weight in co_occurrence.items():
            G.add_edge(pair[0], pair[1], weight=weight)
        net = Network(height="600px", width="100%", notebook=False, directed=False)
        net.from_nx(G)
        net.repulsion(node_distance=200, central_gravity=0.3, spring_length=200, spring_strength=0.05)
        net.save_graph("hashtag_network.html")
        HtmlFile = open("hashtag_network.html",'r',encoding='utf-8').read()
        components.html(HtmlFile, height=600, scrolling=True)

    # -------------------------
    # Tab 9: Recommendations & Insights
    # -------------------------
    with tab9:
        st.markdown("## Automated Recommendations & Insights")

        top_posts = df.sort_values('engagement_rate', ascending=False).head(5)[['date','platform','post_type','topic','engagement_rate']]
        st.markdown("### Top Performing Posts")
        st.dataframe(top_posts)

        low_posts = df.sort_values('engagement_rate').head(5)[['date','platform','post_type','topic','engagement_rate']]
        st.markdown("### Underperforming Posts")
        st.dataframe(low_posts)

        platform_engagement = df.groupby('platform')['engagement_rate'].mean().sort_values(ascending=False)
        st.markdown("### Best Platforms for Engagement")
        st.bar_chart(platform_engagement)

        topic_engagement = df.groupby('topic')['engagement_rate'].mean().sort_values(ascending=False).head(10)
        st.markdown("### Top Topics by Engagement")
        st.bar_chart(topic_engagement)

        all_hashtags = df['hashtags'].dropna().str.split(",", expand=True).stack().str.strip()
        top_hashtags = all_hashtags.value_counts().head(10)
        st.markdown("### Top Hashtags")
        st.bar_chart(top_hashtags)

        st.markdown("### Actionable Recommendations")
        st.write("""
        - Focus on **post types** that show the highest engagement.  
        - Post more frequently on **platforms with higher engagement rates**.  
        - Use trending hashtags and topics identified above.  
        - Review underperforming posts to understand low engagement patterns.  
        - Schedule posts according to engagement trends over time.  
        """)
