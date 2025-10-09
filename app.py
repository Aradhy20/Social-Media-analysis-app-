# social_media_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
import io
import base64
from scipy import stats
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import tempfile
import datetime

st.set_page_config(page_title="TMU Social Media Dashboard", layout="wide")
st.title("TMU Social Media Analytics Dashboard")
st.markdown("Upload one or more CSV files (Facebook/Instagram/etc.). Expected columns: `date`, `post_text`, `likes`, `comments`, `shares`, `followers`, `hashtags`, `platform`, `post_type`, `user`, `mentioned_user`.")
st.markdown("---")

# -----------------------------
# TMU Colors
# -----------------------------
TMU_COLORS = ["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD","#8C564B","#E377C2","#7F7F7F"]

# -----------------------------
# Helpers / caches
# -----------------------------
@st.cache_data
def read_csv_file(f):
    return pd.read_csv(f)

@st.cache_resource
def get_bertopic():
    return BERTopic(verbose=False)

def safe_dt(s):
    return pd.to_datetime(s, errors='coerce')

def make_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color="white", colormap='tab10').generate(text)
    fig = plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    return fig

def df_to_excel_bytes(df_in):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_in.to_excel(writer, index=False, sheet_name="analyzed")
        writer.save()
    return output.getvalue()

def fig_to_image_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

def plotly_fig_to_png_bytes(fig):
    # plotly -> png via to_image (requires kaleido)
    try:
        img_bytes = fig.to_image(format="png")
        return io.BytesIO(img_bytes)
    except Exception:
        # fallback: draw with matplotlib snapshot
        buf = io.BytesIO()
        fig.write_image(buf, format="png")
        buf.seek(0)
        return buf

def generate_pdf_report(filename, meta, kpis, figs, recommendations):
    """
    Create multi-page PDF. 'figs' is list of (title, matplotlib fig or plotly fig)
    """
    packet = io.BytesIO()
    # Use landscape A4
    c = canvas.Canvas(packet, pagesize=landscape(A4))
    width, height = landscape(A4)

    # Cover page
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width/2, height-80, "TMU Social Media Report")
    c.setFont("Helvetica", 12)
    c.drawCentredString(width/2, height-100, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, height-140, f"Files: {', '.join(meta.get('files',[]))}")
    c.drawString(50, height-160, f"Date Range: {meta.get('date_range','All')}")
    c.drawString(50, height-200, "Key KPIs:")
    y = height-220
    for k,v in kpis.items():
        c.drawString(70, y, f"- {k}: {v}")
        y -= 16
    c.showPage()

    # Add each figure as a page
    for title, fig in figs:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, height-40, title)
        # convert fig to image
        if hasattr(fig, "savefig"):
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png', bbox_inches='tight')
            img_buf.seek(0)
        else:
            # assume plotly
            try:
                img_buf = plotly_fig_to_png_bytes(fig)
            except Exception:
                img_buf = io.BytesIO()
        img = ImageReader(img_buf)
        # draw centered
        img_w = width - 100
        img_h = height - 120
        c.drawImage(img, 50, 60, width=img_w, height=img_h, preserveAspectRatio=True, anchor='c')
        c.showPage()

    # Recommendations page
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height-40, "Recommendations & Insights")
    c.setFont("Helvetica", 12)
    y = height-80
    for r in recommendations:
        if y < 80:
            c.showPage()
            y = height-80
        c.drawString(50, y, "- " + r)
        y -= 18

    c.save()
    packet.seek(0)
    return packet.getvalue()

# -----------------------------
# Multi-file upload
# -----------------------------
uploaded_files = st.file_uploader("Upload CSV files", type=['csv'], accept_multiple_files=True)
if not uploaded_files:
    st.info("Please upload one or more CSV files to begin.")
    st.stop()

dfs = []
errs = []
for up in uploaded_files:
    try:
        tmp = read_csv_file(up)
        tmp['source_file'] = up.name
        dfs.append(tmp)
    except Exception as e:
        errs.append((up.name, str(e)))
if errs:
    for n,e in errs:
        st.error(f"Failed to read {n}: {e}")
if not dfs:
    st.error("No valid CSVs uploaded.")
    st.stop()

df = pd.concat(dfs, ignore_index=True)

# -----------------------------
# Data cleaning & enrichment
# -----------------------------
# ensure necessary columns exist
needed = ["date","post_text","likes","comments","shares","followers","hashtags","platform","post_type","user","mentioned_user"]
for col in needed:
    if col not in df.columns:
        df[col] = np.nan

# normalize
df['date'] = safe_dt(df['date'])
df = df[~df['date'].isna()].copy()
for ncol in ["likes","comments","shares","followers"]:
    df[ncol] = pd.to_numeric(df[ncol], errors='coerce').fillna(0)

df['post_text'] = df['post_text'].fillna("").astype(str)
df['hashtags'] = df['hashtags'].fillna("").astype(str)
df['platform'] = df['platform'].fillna("").astype(str)
df['post_type'] = df['post_type'].fillna("Other").astype(str).str.capitalize()
df['user'] = df['user'].fillna("").astype(str)
df['mentioned_user'] = df['mentioned_user'].fillna("").astype(str)

# engagement metrics
df['engagement_score'] = df['likes'] + 2*df['comments'] + 3*df['shares']
df['engagement_rate'] = df.apply(lambda r: (r['engagement_score']/r['followers']*100) if r['followers']>0 else 0, axis=1)

# virality z-score
df['engagement_z'] = df.groupby('platform')['engagement_score'].transform(lambda x: stats.zscore(x.fillna(0)))
df['engagement_z'] = df['engagement_z'].fillna(stats.zscore(df['engagement_score'].fillna(0)))
df['is_viral'] = df['engagement_z'] > 2

# -----------------------------
# Sidebar global filters
# -----------------------------
st.sidebar.header("Filters")
plat_opts = sorted(df['platform'].unique())
sel_plats = st.sidebar.multiselect("Platforms", options=plat_opts, default=plat_opts)
pt_opts = sorted(df['post_type'].unique())
sel_pt = st.sidebar.multiselect("Post Types", options=pt_opts, default=pt_opts)
min_date, max_date = df['date'].min(), df['date'].max()
sel_dates = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

df = df[df['platform'].isin(sel_plats) & df['post_type'].isin(sel_pt)]
df = df[(df['date'] >= pd.to_datetime(sel_dates[0])) & (df['date'] <= pd.to_datetime(sel_dates[1]))]

# -----------------------------
# Topic modeling (BERTopic) - cached
# -----------------------------
@st.cache_resource
def fit_topic_model(docs):
    model = BERTopic(verbose=False)
    topics, probs = model.fit_transform(docs)
    return model, topics, probs

with st.spinner("Running topic modeling..."):
    try:
        docs = df['post_text'].astype(str).tolist()
        topic_model, topics, probs = fit_topic_model(docs)
        df['topic'] = topics
    except Exception as e:
        st.warning(f"BERTopic error: {e}")
        df['topic'] = df.get('topic', pd.Series(np.arange(len(df))))

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs([
    "Overview / KPIs",
    "Engagement Trends",
    "Sentiment Analysis",
    "Post Clustering",
    "Engagement Forecast",
    "Topics / Unsupervised",
    "Networks",
    "Hashtag Co-occurrence",
    "Recommendations & PDF"
])

# -----------------------------
# Tab 1: Overview / KPIs
# -----------------------------
with tabs[0]:
    st.header("Overview & KPIs")
    cols = st.columns(5)
    cols[0].metric("Total Likes", int(df['likes'].sum()))
    cols[1].metric("Total Comments", int(df['comments'].sum()))
    cols[2].metric("Total Shares", int(df['shares'].sum()))
    cols[3].metric("Avg Engagement Score", round(df['engagement_score'].mean(),2))
    cols[4].metric("Avg Engagement Rate (%)", round(df['engagement_rate'].mean(),2))

    st.markdown("**File summary**")
    st.dataframe(df.groupby('source_file')[['likes','comments','shares','engagement_score']].sum().reset_index())

    st.markdown("**Top viral posts**")
    if df['is_viral'].any():
        st.dataframe(df[df['is_viral']].sort_values('engagement_z', ascending=False)[['date','platform','post_type','engagement_score','engagement_z','post_text']].head(10))
    else:
        st.info("No viral posts in selection (z>2).")

# -----------------------------
# Tab 2: Engagement Trends
# -----------------------------
with tabs[1]:
    st.header("Engagement Trends")
    eng = df.groupby('date')['engagement_score'].sum().reset_index()
    fig = px.line(eng, x='date', y='engagement_score', title="Total Engagement Over Time", color_discrete_sequence=[TMU_COLORS[0]])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Engagement rate by platform")
    plat = df.groupby(['date','platform'])['engagement_rate'].mean().reset_index()
    fig2 = px.line(plat, x='date', y='engagement_rate', color='platform', title="Engagement Rate by Platform", color_discrete_sequence=TMU_COLORS)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("Top posting hours")
    hours = df.groupby(df['date'].dt.hour)['engagement_score'].mean().sort_values(ascending=False).head(6)
    st.bar_chart(hours)

# -----------------------------
# Tab 3: Sentiment
# -----------------------------
with tabs[2]:
    st.header("Sentiment Analysis")
    if 'sentiment' not in df.columns:
        df['sentiment'] = df['post_text'].apply(lambda x: "Positive" if TextBlob(x).sentiment.polarity>0.1 else ("Negative" if TextBlob(x).sentiment.polarity<-0.1 else "Neutral"))
    sc = df['sentiment'].value_counts()
    fig_s = px.pie(names=sc.index, values=sc.values, title="Sentiment Distribution", color_discrete_sequence=TMU_COLORS)
    st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("Hashtags Wordcloud")
    wc_fig = make_wordcloud(" ".join(df['hashtags'].astype(str)))
    st.pyplot(wc_fig)

# -----------------------------
# Tab 4: Clustering
# -----------------------------
with tabs[3]:
    st.header("Clustering (KMeans + t-SNE)")
    if df.shape[0] >= 5:
        features = ['likes','comments','shares','engagement_score']
        X = df[features].fillna(0).values
        k = min(6, max(2, int(np.sqrt(len(df)))))
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['cluster'] = kmeans.fit_predict(X)
        tsne = TSNE(n_components=2, random_state=42)
        res = tsne.fit_transform(X)
        df['tsne_1'], df['tsne_2'] = res[:,0], res[:,1]
        figc = px.scatter(df, x='tsne_1', y='tsne_2', color='cluster', hover_data=['post_text','engagement_score'], title="t-SNE clusters", color_discrete_sequence=TMU_COLORS)
        st.plotly_chart(figc, use_container_width=True)
        st.markdown("Cluster summary")
        st.dataframe(df.groupby('cluster')[features].mean().round(2))
    else:
        st.info("Not enough posts for clustering.")

# -----------------------------
# Tab 5: Forecast
# -----------------------------
with tabs[4]:
    st.header("Engagement Forecast (Prophet)")
    df_fore = df.groupby('date')['engagement_score'].sum().reset_index().rename(columns={'date':'ds','engagement_score':'y'})
    if df_fore.shape[0] >= 2:
        try:
            m = Prophet()
            m.fit(df_fore)
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)
            figf = px.line(forecast, x='ds', y='yhat', title="30-day Engagement Forecast")
            st.plotly_chart(figf, use_container_width=True)
        except Exception as e:
            st.warning(f"Prophet error: {e}")
    else:
        st.info("Not enough data points to forecast.")

    st.markdown("Platform-specific forecast")
    plat_choice = st.selectbox("Platform", options=["Overall"] + plat_opts)
    if plat_choice != "Overall":
        df_pf = df[df['platform'] == plat_choice].groupby('date')['engagement_score'].sum().reset_index().rename(columns={'date':'ds','engagement_score':'y'})
        if df_pf.shape[0] >= 2:
            try:
                m2 = Prophet()
                m2.fit(df_pf)
                future2 = m2.make_future_dataframe(periods=30)
                f2 = m2.predict(future2)
                st.plotly_chart(px.line(f2, x='ds', y='yhat', title=f"{plat_choice} forecast"), use_container_width=True)
            except Exception as e:
                st.warning(f"Prophet error for {plat_choice}: {e}")
        else:
            st.info("Not enough data for selected platform.")

# -----------------------------
# Tab 6: Topics
# -----------------------------
with tabs[5]:
    st.header("Topics (BERTopic)")
    try:
        tcounts = df['topic'].value_counts().reset_index()
        tcounts.columns = ['topic','count']
        st.plotly_chart(px.bar(tcounts.head(20), x='topic', y='count', title="Topic counts"), use_container_width=True)
        top_topic = df['topic'].value_counts().idxmax()
        st.markdown(f"Top topic id: {top_topic}")
        st.dataframe(df[df['topic'] == top_topic][['date','platform','post_type','engagement_score','post_text']].head(10))
    except Exception as e:
        st.warning(f"Topic error: {e}")

# -----------------------------
# Tab 7: Networks (mentions)
# -----------------------------
with tabs[6]:
    st.header("User Mentions Network (interactive)")
    top_n = st.slider("Top N users", min_value=5, max_value=100, value=25, step=5)
    topics_net = sorted(df['topic'].dropna().unique())
    sel_topics_net = st.multiselect("Topics", options=topics_net, default=topics_net)
    sel_pt_net = st.multiselect("Post Types", options=pt_opts, default=pt_opts)
    df_net = df[df['topic'].isin(sel_topics_net) & df['post_type'].isin(sel_pt_net)]
    top_users = df_net.groupby('user')['engagement_score'].sum().sort_values(ascending=False).head(top_n).index.tolist()
    df_net = df_net[df_net['user'].isin(top_users)]
    G = nx.DiGraph()
    for _,r in df_net.iterrows():
        src = r['user']; targets = [t.strip() for t in str(r['mentioned_user']).split(",") if t.strip()]
        for t in targets:
            if t in top_users:
                G.add_edge(src, t)
    if G.number_of_nodes() == 0:
        st.info("No mention network to show for selection.")
    else:
        net = Network(height="600px", width="100%", notebook=False, directed=True)
        net.from_nx(G)
        net.repulsion(node_distance=200)
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        net.save_graph(tmpf.name)
        HtmlFile = open(tmpf.name, 'r', encoding='utf-8').read()
        components.html(HtmlFile, height=600, scrolling=True)

# -----------------------------
# Tab 8: Hashtag co-occurrence
# -----------------------------
with tabs[7]:
    st.header("Hashtag Co-Occurrence Network")
    top_h = st.slider("Top N hashtags", min_value=10, max_value=200, value=50, step=10)
    topics_ht = sorted(df['topic'].dropna().unique())
    sel_topics_ht = st.multiselect("Topics", options=topics_ht, default=topics_ht)
    sel_pt_ht = st.multiselect("Post Types", options=pt_opts, default=pt_opts)
    df_ht = df[df['topic'].isin(sel_topics_ht) & df['post_type'].isin(sel_pt_ht)]
    all_ht = df_ht['hashtags'].dropna().str.split(",", expand=True).stack().str.strip()
    top_ht = all_ht.value_counts().head(top_h).index.tolist()
    co = {}
    for tags in df_ht['hashtags'].dropna():
        tags_list = [t.strip() for t in tags.split(",") if t.strip() in top_ht]
        for i in range(len(tags_list)):
            for j in range(i+1, len(tags_list)):
                pair = tuple(sorted([tags_list[i], tags_list[j]]))
                co[pair] = co.get(pair,0) + 1
    if not co:
        st.info("No hashtag co-occurrence data for selection.")
    else:
        Ght = nx.Graph()
        for p,w in co.items():
            Ght.add_edge(p[0], p[1], weight=w)
        net = Network(height="600px", width="100%", notebook=False, directed=False)
        net.from_nx(Ght)
        net.repulsion(node_distance=200)
        tmpf2 = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        net.save_graph(tmpf2.name)
        HtmlFile = open(tmpf2.name, 'r', encoding='utf-8').read()
        components.html(HtmlFile, height=600, scrolling=True)

# -----------------------------
# Tab 9: Recommendations & PDF
# -----------------------------
with tabs[8]:
    st.header("Recommendations & Export")

    # Platform comparison
    platform_summary = df.groupby('platform').agg(
        total_posts=('post_text','count'),
        avg_engagement_rate=('engagement_rate','mean'),
        avg_engagement_score=('engagement_score','mean')
    ).sort_values('avg_engagement_rate', ascending=False)
    st.subheader("Platform comparison")
    st.dataframe(platform_summary.round(2))

    # top post types
    st.subheader("Top post types by engagement rate")
    top_types = df.groupby('post_type')['engagement_rate'].mean().sort_values(ascending=False)
    st.bar_chart(top_types)

    # top topics & hashtags
    st.subheader("Top topics & hashtags")
    st.dataframe(df['topic'].value_counts().head(20).rename_axis('topic').reset_index(name='count'))
    st.dataframe(df['hashtags'].dropna().str.split(",", expand=True).stack().str.strip().value_counts().head(20).rename_axis('hashtag').reset_index(name='count'))

    # best posting hours
    st.subheader("Best posting hours (avg engagement score)")
    hours = df.groupby(df['date'].dt.hour)['engagement_score'].mean().sort_values(ascending=False)
    st.write(list(hours.head(6).index))

    # viral examples
    st.subheader("Viral posts examples")
    st.dataframe(df.sort_values('engagement_z', ascending=False)[['date','platform','post_type','engagement_score','engagement_z','post_text']].head(10))

    # Generate narrative recommendations
    recs = []
    if not platform_summary.empty:
        best_platform = platform_summary['avg_engagement_rate'].idxmax()
        recs.append(f"Focus resources on {best_platform} — highest avg engagement rate ({platform_summary.loc[best_platform,'avg_engagement_rate']:.2f}%).")
    if not top_types.empty:
        best_pt = top_types.idxmax()
        recs.append(f"Prioritize content type: {best_pt} — highest avg engagement rate among post types.")
    if not hours.empty:
        best_hour = int(hours.idxmax())
        recs.append(f"Post around {best_hour}:00 — historically highest engagement hour.")
    neg_pct = round((df['sentiment']=="Negative").mean()*100,1) if 'sentiment' in df.columns else 0
    if neg_pct > 20:
        recs.append(f"Negative sentiment is {neg_pct}%. Consider reviewing customer feedback and improving response tone.")
    if df['is_viral'].sum()>0:
        recs.append("Viral posts detected — analyze top viral content and replicate its format/topics.")

    st.subheader("Actionable Recommendations")
    for r in recs:
        st.write("- " + r)

    # Prepare PDF & Excel exports
    st.markdown("---")
    st.subheader("Export Reports")

    # prepare KPI summary and figures to include in PDF
    kpis = {
        "Total Likes": int(df['likes'].sum()),
        "Total Comments": int(df['comments'].sum()),
        "Total Shares": int(df['shares'].sum()),
        "Avg Engagement Score": round(df['engagement_score'].mean(),2),
        "Avg Engagement Rate (%)": round(df['engagement_rate'].mean(),2),
    }

    figs_for_pdf = []
    # capture engagement fig
    try:
        figs_for_pdf.append(("Engagement Over Time", fig))
    except:
        pass
    try:
        figs_for_pdf.append(("Engagement Rate by Platform", fig2))
    except:
        pass
    try:
        figs_for_pdf.append(("Sentiment Distribution", fig_s))
    except:
        pass
    # wordcloud matplotlib fig
    try:
        figs_for_pdf.append(("Hashtags Wordcloud", wc_fig))
    except:
        pass

    # meta
    meta = {
        "files": [f.name for f in uploaded_files],
        "date_range": f"{sel_dates[0]} to {sel_dates[1]}"
    }

    if st.button("Generate PDF Report"):
        with st.spinner("Creating PDF..."):
            try:
                pdf_bytes = generate_pdf_report(
                    filename="report.pdf",
                    meta=meta,
                    kpis=kpis,
                    figs=figs_for_pdf,
                    recommendations=recs
                )
                date_label = f"{sel_dates[0]}_{sel_dates[1]}"
                fname = f"Social_Media_Report_{date_label}.pdf"
                st.download_button("Download PDF Report", data=pdf_bytes, file_name=fname, mime="application/pdf")
            except Exception as e:
                st.error(f"Failed to generate PDF: {e}")

    # Excel download of full cleaned dataset
    excel_bytes = df_to_excel_bytes = df_to_excel_bytes = df_to_excel_bytes if False else None
    try:
        excel_data = df_to_excel_bytes(df)
        st.download_button("Download Excel (cleaned & analyzed)", data=excel_data, file_name=f"social_media_analyzed_{sel_dates[0]}_{sel_dates[1]}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        # fallback simple excel bytes
        try:
            excel_data = df_to_excel_bytes(df)
            st.download_button("Download Excel (cleaned & analyzed)", data=excel_data, file_name=f"social_media_analyzed_{sel_dates[0]}_{sel_dates[1]}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"Excel export failed: {e}")

st.markdown("---")
st.caption("TMU Social Media Dashboard — includes unsupervised topic modeling (BERTopic), clustering, forecasting (Prophet), network visualizations, and PDF export.")
