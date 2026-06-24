import os
import torch

# --- BUG FIX FOR STREAMLIT + PYTORCH ---
# 1. Neutralize the path inspection that causes the RuntimeError
torch.classes.__path__ = [] 
# 2. Optionally disable Streamlit's file watcher (belt-and-suspenders approach)
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
# ---------------------------------------

import concurrent.futures
import urllib.parse
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from transformers import pipeline

# Set Page Config
st.set_page_config(
    page_title="Financial Sentiment Engine", page_icon="📈", layout="wide"
)

# --- CUSTOM HIGH & MID CAP LIST ---
# This list is used to quickly filter companies without making external API calls
HIGH_MID_CAP_COMPANIES = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "ITC", "SBIN",
    "BHARTIARTL", "HINDUNILVR", "BAJFINANCE", "LT", "KOTAKBANK", "AXISBANK",
    "HCLTECH", "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO",
    "TATASTEEL", "WIPRO", "ONGC", "NTPC", "POWERGRID", "M&M", "ADANIENT",
    "ADANIPORTS", "HINDALCO", "JSWSTEEL", "GRASIM", "TECHM", "BAJAJFINSV",
    "DIVISLAB", "TATAMOTORS", "COALINDIA", "SBILIFE", "HDFCLIFE", "CIPLA",
    "APOLLOHOSP", "DRREDDY", "BRITANNIA", "EICHERMOT", "INDUSINDBK",
    "TATACHM", "TATAPOWER", "TVSMOTOR", "HEROMOTOCO", "BAJAJ-AUTO", "BHEL",
    "GAIL", "BPCL", "IOC", "PIDILITIND", "HAVELLS", "GODREJCP", "DABUR",
    "MARICO", "COLPAL", "MCDOWELL-N", "UPL", "AMBUJACEM", "SHREECEM",
    "SRF", "TRENT", "VEDL", "INDIGO", "ZOMATO", "PAYTM", "NYKAA", "DMART",
    "HAL", "BEL", "SIEMENS", "ABB", "CUMMINSIND", "LTIM", "PERSISTENT",
    "COFORGE", "TATACOMM", "TATAELXSI", "VOLTAS", "DIXON", "POLYCAB",
    "ASTRAL", "SUPREMEIND", "MUTHOOTFIN", "CHOLAFIN", "PNB", "BOB", "CANBK",
    "UNIONBANK", "IDFCFIRSTB", "FEDERALBNK", "BANDHANBNK", "AUROPHARMA",
    "LUPIN", "TORNTPHARM", "ALKEM", "MAXHEALTH", "PAGEIND", "BATAINDIA"
]
# ----------------------------------

# --- 1. CACHED MODEL INITIALIZATION ---
# This ensures the model is loaded into memory only ONCE when the app boots up
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

sentiment_analyzer = load_sentiment_model()

def get_financial_sentiment(text):
    """Calculates text sentiment using loaded FinBERT."""
    if not isinstance(text, str) or not text.strip():
        return "neutral"
    try:
        result = sentiment_analyzer(text)[0]
        return result["label"]
    except Exception:
        return "neutral"

# --- 2. CORE UTILITY SCRAPER FUNCTION ---
def fetch_and_score_company(company_name, filter_cap=False):
    """Fetches, filters, and analyzes news for a single company identifier."""
    
    # --- ADDED MARKET CAP FILTER (Using custom list) ---
    if filter_cap:
        # Check if the company exists in our predefined list (case-insensitive check)
        if company_name.upper() not in HIGH_MID_CAP_COMPANIES:
            return None  # Skip companies not in our predefined list
    # -------------------------------

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    query = f"{company_name} stock OR earnings OR NSE"
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Read XML response string
        df_news = pd.read_xml(response.content, xpath=".//item")

        if df_news.empty or "title" not in df_news.columns:
            return None

        # Clean and case-insensitive target keyword string matching filter
        df_news = df_news.dropna(subset=["title"])
        df_news["title"] = df_news["title"].astype(str)

        company_lower = str(company_name).lower()
        mention_mask = df_news["title"].str.lower().str.contains(
            company_lower, regex=False
        )
        df_news = df_news[mention_mask].copy()

        if df_news.empty:
            return None

        # Process text arrays
        df_news["company_name"] = company_name
        df_news["sentiment"] = df_news["title"].apply(get_financial_sentiment)

        # Calculate Net Score Aggregations
        total_articles = len(df_news)
        pos_count = len(df_news[df_news["sentiment"] == "positive"])
        neg_count = len(df_news[df_news["sentiment"] == "negative"])
        net_sentiment = (pos_count - neg_count) / total_articles

        cols_to_keep = [
            "company_name",
            "title",
            "sentiment",
            "link",
            "pubDate",
            "source",
        ]
        existing_cols = [c for c in cols_to_keep if c in df_news.columns]

        return {
            "summary": {
                "Company": company_name,
                "Total Articles": total_articles,
                "Net Sentiment": round(net_sentiment, 2),
                "Positive": pos_count,
                "Negative": neg_count,
                "Neutral": total_articles - (pos_count + neg_count),
            },
            "news_df": df_news[existing_cols],
        }
    except Exception:
        return None

# --- 3. STREAMLIT INTERFACE SIDEBAR & NAVIGATION ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose Analysis Mode",
    ["Single Company Deep-Dive", "Bulk Screen Market"],
)

# Load underlying local target CSV data asset safely
try:
    data_companies = pd.read_csv("company.csv")
    company_list = data_companies["company_name"].dropna().tolist()
except Exception as e:
    st.error(
        f"Could not load 'company.csv'. Make sure it is in the same directory. Error: {e}"
    )
    st.stop()

# --- SECTION A: SINGLE COMPANY MODE ---
if app_mode == "Single Company Deep-Dive":
    st.title("📊 Single Company Sentiment Deep-Dive")
    st.write(
        "Select a specific tracking asset ticker to fetch relevant real-time contextual news items."
    )

    selected_company = st.selectbox("Select Company Target", company_list)

    if st.button("Run Analytics Engine", type="primary"):
        with st.spinner(f"Scraping and analyzing news for {selected_company}..."):
            # Disable Market Cap filter for manual single company scans
            result = fetch_and_score_company(selected_company, filter_cap=False)

            if result is None:
                st.warning(
                    f"No explicitly verified tracking articles found mentioning '{selected_company}' inside the titles today."
                )
            else:
                summary = result["summary"]
                news_df = result["news_df"]

                # 1. Metric Display row cards
                st.subheader(f"Sentiment Summary Metrics: {selected_company}")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Articles Evaluated", summary["Total Articles"])

                # Determine dynamic delta color styling based on thresholds
                net_score = summary["Net Sentiment"]
                if net_score > 0.4:
                    status = "Bullish"
                elif net_score < -0.2:
                    status = "Bearish"
                else:
                    status = "Neutral"

                col2.metric("Net Sentiment Score", f"{net_score:.2f}", status)
                col3.metric("Positive HeadlinesCount", summary["Positive"])
                col4.metric("Negative Headlines Count", summary["Negative"])

                # 2. Layout Distribution Visualization Split Plots
                st.write("---")
                st.subheader("Data Distribution Plots")
                fig, ax = plt.subplots(
                    figsize=(6, 3)
                )  # Managed dimensions for clean look
                sentiment_labels = ["positive", "neutral", "negative"]
                counts = [
                    summary["Positive"],
                    summary["Neutral"],
                    summary["Negative"],
                ]

                # Map clear structural UI colors safely
                palette_colors = ["#2ca02c", "#7f7f7f", "#d62728"]
                sns.barplot(
                    x=sentiment_labels,
                    y=counts,
                    palette=palette_colors,
                    ax=ax,
                    hue=sentiment_labels,
                    legend=False,
                )
                ax.set_ylabel("Article Counts")
                ax.set_title("Headline Classification Spread")
                st.pyplot(fig)

                # 3. Clean Categorical Raw Interactive Data Frame Output Layout View
                st.write("---")
                st.subheader("Scraped Match Data Feed Logs")
                st.dataframe(
                    news_df,
                    column_config={
                        "link": st.column_config.LinkColumn(
                            "Source News Article URL"
                        )
                    },
                    use_container_width=True,
                )

# --- SECTION B: BULK SCREEN MARKET MODE ---
elif app_mode == "Bulk Screen Market":
    st.title("🔍 Bulk Market Sentiment Screener")
    st.write(
        "Processes multiple companies simultaneously to surface high-sentiment opportunities. You can exclusively scan **High and Mid-cap stocks** to filter out low-volume noise."
    )

    col1, col2 = st.columns(2)
    with col1:
        scan_limit = int(st.number_input("Total companies to scan from CSV:", min_value=10, max_value=2000, value=200, step=50))
    with col2:
        st.write("")
        st.write("")
        filter_cap = st.checkbox("Scan Predefined High & Mid-Cap ONLY", value=True, help="Filters using a built-in Python list of top NSE companies instead of scanning all small caps.")

    target_subset = company_list[:scan_limit]
    st.info(f"Ready to evaluate top {len(target_subset)} companies. (Small caps will be dynamically skipped if the filter is checked).")

    if st.button("Start Bulk Screen Processing", type="primary"):
        all_summaries = []
        all_news_collections = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Execute ThreadPoolExecutor for high throughput concurrent web network I/O
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            status_text.text("Dispatched simultaneous worker queries...")
            future_to_company = {
                executor.submit(fetch_and_score_company, comp, filter_cap): comp
                for comp in target_subset
            }

            completed = 0
            for future in concurrent.futures.as_completed(future_to_company):
                comp_name = future_to_company[future]
                completed += 1

                # Update operational interactive feedback status display cleanly
                percent_done = int((completed / len(target_subset)) * 100)
                progress_bar.progress(percent_done)
                status_text.text(
                    f"Processing: ({completed}/{len(target_subset)}) - Evaluated {comp_name}"
                )

                try:
                    res = future.result()
                    if res is not None:
                        all_summaries.append(res["summary"])
                        all_news_collections.append(res["news_df"])
                except Exception:
                    pass

        status_text.success(f"Market Screening Complete! Found {len(all_summaries)} companies meeting all criteria.")

        if not all_summaries:
            st.error(
                "No target corporate data matches recovered. All scanned companies were either small-cap or had no relevant news."
            )
        else:
            df_summaries_all = pd.DataFrame(all_summaries)

            # Isolate only highly positive entries according to initial criteria thresholds
            df_bullish = df_summaries_all[df_summaries_all["Net Sentiment"] > 0.4]

            # Section Layout Render Containers Splits
            st.write("---")
            st.subheader("🔥 High Sentiment Bullish Companies (> 0.4)")

            if df_bullish.empty:
                st.info(
                    "No listed entities crossed the strict 0.4 net positive momentum threshold parameter index settings."
                )
            else:
                st.dataframe(
                    df_bullish.sort_values(
                        by="Net Sentiment", ascending=False
                    ).reset_index(drop=True),
                    use_container_width=True,
                )

                # Download Options Data Trigger Link Actions
                csv_download = df_bullish.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Bullish Report CSV",
                    data=csv_download,
                    file_name="screened_bullish_opportunities.csv",
                    mime="text/csv",
                )

            st.write("---")
            st.subheader("📋 Complete Screen Log Overview (Valid Assets Only)")
            st.dataframe(
                df_summaries_all.sort_values(by="Net Sentiment", ascending=False),
                use_container_width=True,
            )
