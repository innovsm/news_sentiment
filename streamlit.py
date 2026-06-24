import concurrent.futures
import os
import urllib.parse
import pandas as pd
import requests

# Global variable inside each worker process to store its local model instance
_worker_sentiment_analyzer = None


def init_worker():
    """Initializes the FinBERT model exactly once per CPU core/worker process."""
    global _worker_sentiment_analyzer
    # Import inside the worker to keep the main process lightweight
    from transformers import pipeline

    if _worker_sentiment_analyzer is None:
        _worker_sentiment_analyzer = pipeline(
            "sentiment-analysis", model="ProsusAI/finbert"
        )


def process_single_company(company):
    """The task function that a single process worker will execute for one company."""
    global _worker_sentiment_analyzer

    # Standard headers to prevent 403 blocks
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    query = f"{company} stock OR earnings OR NSE"
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        # 1. Fetch RSS Feed
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # 2. Read XML items
        df_news = pd.read_xml(response.content, xpath=".//item")

        if df_news.empty or "title" not in df_news.columns:
            return None

        # 3. Case-insensitive Title Filtering
        df_news = df_news.dropna(subset=["title"])
        df_news["title"] = df_news["title"].astype(str)

        company_lower = str(company).lower()
        mention_mask = df_news["title"].str.lower().str.contains(
            company_lower, regex=False
        )
        df_news = df_news[mention_mask].copy()

        if df_news.empty:
            return None

        # 4. Attach Company Name
        df_news["company_name"] = company

        # 5. Run FinBERT Sentiment
        def get_sentiment(text):
            try:
                return _worker_sentiment_analyzer(text)[0]["label"]
            except Exception:
                return "neutral"

        df_news["sentiment"] = df_news["title"].apply(get_sentiment)

        # 6. Calculate Metrics
        total_articles = len(df_news)
        pos_count = len(df_news[df_news["sentiment"] == "positive"])
        neg_count = len(df_news[df_news["sentiment"] == "negative"])
        net_sentiment = (pos_count - neg_count) / total_articles

        if net_sentiment > 0.4:
            outlook = "Bullish"
        elif net_sentiment < -0.2:
            outlook = "Bearish"
        else:
            outlook = "Neutral / Mixed"

        # Limit tracking columns for clean output
        cols_to_keep = [
            "company_name",
            "title",
            "sentiment",
            "link",
            "pubDate",
            "source",
        ]
        existing_cols = [c for c in cols_to_keep if c in df_news.columns]

        # Return both summary data and the matching headlines
        summary = {
            "Company": company,
            "Total Articles": total_articles,
            "Net Sentiment": round(net_sentiment, 2),
            "Outlook": outlook,
        }

        return {"summary": summary, "news_df": df_news[existing_cols]}

    except Exception as e:
        # Gracefully handle single company errors so the entire pool doesn't crash
        return None


# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Load your company dataset
    data_companies = pd.read_csv("company.csv")
    
    # --- UPDATED LINE ---
    # Extract specifically the 'company_name' column from the CSV based on the image
    company_list = data_companies["company_name"].dropna().head(100).tolist()
    # --------------------

    all_results = []
    all_news_dfs = []
    bullish_companies = []

    # Decide how many parallel workers to run (Default: total CPU cores minus 1)
    max_workers = max(1, 2)
    print(
        f"Starting parallel pipeline with {max_workers} workers for {len(company_list)} companies..."
    )

    # 2. Launch the Process Pool
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers, initializer=init_worker
    ) as executor:
        # Map the function over our list of companies
        results = executor.map(process_single_company, company_list)

        # 3. Collect outcomes as they finish
        for res in results:
            if res is not None:
                summary_data = res["summary"]
                all_results.append(summary_data)
                all_news_dfs.append(res["news_df"])

                # Separate tracking for bullish output
                if summary_data["Outlook"] == "Bullish":
                    bullish_companies.append(
                        {
                            "Company": summary_data["Company"],
                            "Net Sentiment": summary_data["Net Sentiment"],
                        }
                    )
                print(
                    f"Finished: {summary_data['Company']} -> {summary_data['Outlook']} ({summary_data['Net Sentiment']})"
                )

    # 4. Save compiled DataFrames to Disk
    print("\n" + "=" * 40 + "\nPROCESSING COMPLETE\n" + "=" * 40)

    if all_news_dfs:
        df_master_news = pd.concat(all_news_dfs, ignore_index=True)
        df_master_news.to_csv("all_filtered_news.csv", index=False)
        print(f"Saved {len(df_master_news)} total headlines to 'all_filtered_news.csv'")

    df_bullish = pd.DataFrame(bullish_companies)
    print("\n--- Bullish Companies Found ---")
    if not df_bullish.empty:
        print(df_bullish.to_string(index=False))
        df_bullish.to_csv("bullish_companies.csv", index=False)
    else:
        print("No companies met the > 0.4 bullish threshold today.")
