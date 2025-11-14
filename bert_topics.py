import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


def main():
    # 1. Load scraped articles
    df = pd.read_csv("outseer_articles.csv")

    # --- DATE HANDLING ---
    # Try date first, else fall back to published; if neither exists, create empty date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "published" in df.columns:
        df["date"] = pd.to_datetime(df["published"], errors="coerce")
    else:
        df["date"] = pd.NaT  # still give Streamlit a column to work with

    # 2. Build combined document text (title + full_text)
    df["doc"] = (
        df["title"].fillna("").astype(str)
        + ". "
        + df["full_text"].fillna("").astype(str)
    )

    # 3. Build stopword set (for cleaning docs + later filtering keywords)
    stop_words = set(ENGLISH_STOP_WORDS)
    custom_words = {
        "the", "to", "and", "of", "in", "for", "is", "that",
        "with", "on", "as", "by", "it", "this", "these", "those",
        "are", "be", "from", "at", "an", "or", "we", "you", "our"
    }
    stop_words.update(custom_words)

    # 4. Clean docs by removing stopwords BEFORE modeling
    df["doc_clean"] = df["doc"].str.lower().str.split().apply(
        lambda tokens: " ".join(t for t in tokens if t not in stop_words)
    )
    docs_clean = df["doc_clean"].tolist()

    # 5. Load embedding model
    print("Loading sentence transformer model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # IMPORTANT: use built-in English stopwords here (string, not a set)
    vectorizer_model = CountVectorizer(stop_words="english")

    # 6. Fit BERTopic (NO timestamps arg – your version doesn’t support it)
    print("Fitting BERTopic model on documents...")
    topic_model = BERTopic(
        embedding_model=embed_model,
        vectorizer_model=vectorizer_model,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(docs_clean)

    # 7. Attach topics and probabilities back to each article
    df["topic"] = topics
    df["probability"] = probs
    # NOTE: 'date' stays in this CSV for time-based plots
    df.to_csv("outseer_articles_with_topics.csv", index=False)
    print("Saved article-level topics → outseer_articles_with_topics.csv")

    # 8. Topic overview
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv("bertopic_topics_overview.csv", index=False)
    print("Saved topic overview → bertopic_topics_overview.csv")

    # 9. Extract keywords + weights for each topic
    keyword_rows = []
    for topic_id in topic_info["Topic"]:
        if topic_id == -1:  # skip outlier topic
            continue
        words = topic_model.get_topic(topic_id)  # list of (word, weight)
        for rank, (word, weight) in enumerate(words, start=1):
            keyword_rows.append(
                {
                    "topic": topic_id,
                    "rank": rank,
                    "keyword": word,
                    "weight": weight,
                }
            )

    kw_df = pd.DataFrame(keyword_rows)

    # 10. Remove stopwords from keyword table
    kw_df["keyword"] = kw_df["keyword"].str.lower().str.strip()
    kw_df = kw_df[~kw_df["keyword"].isin(stop_words)]

    # 11. Keep ONLY fraud-related terms
    fraud_pattern = (
        r"(fraud|frauds|fraudster|fraudsters|"
        r"scam|scams|phish|phishing|"
        r"mule|mules|launder|laundering|"
        r"chargeback|chargebacks|"
        r"breach|breaches|"
        r"identity|id theft|theft)"
    )
    kw_df = kw_df[kw_df["keyword"].str.contains(fraud_pattern, case=False, regex=True)]

    kw_df.to_csv("bertopic_keywords_weights.csv", index=False)
    print("✅ Fraud-only keyword weights → bertopic_keywords_weights.csv")


if __name__ == "__main__":
    main()
