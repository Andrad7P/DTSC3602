import pandas as pd
import re

# 1. Load keywords and article-level topics
kw = pd.read_csv("bertopic_keywords_weights_cleaned.csv")
articles = pd.read_csv("outseer_articles_with_topics.csv", parse_dates=["date"])

# make a lowercase text field to search
articles["doc_clean"] = articles["doc"].astype(str).str.lower()
kw["keyword"] = kw["keyword"].str.lower()

def get_keyword_dates(row):
    # Filter articles with same topic
    topic_mask = articles["topic"] == row["topic"]

    # Look for the keyword as a whole word in the article text
    pattern = r"\b" + re.escape(row["keyword"]) + r"\b"
    text_mask = articles["doc_clean"].str.contains(pattern, na=False)

    hits = articles.loc[topic_mask & text_mask, "date"]

    if hits.empty:
        return pd.Series([pd.NaT, pd.NaT, 0])

    return pd.Series([hits.min(), hits.max(), len(hits)])

# 2. Apply per (topic, keyword)
kw[["first_date", "last_date", "doc_count"]] = kw.apply(get_keyword_dates, axis=1)

# 3. Save new file
kw.to_csv("bertopic_keywords_weights_with_dates.csv", index=False)
print("✅ Saved keyword weights with dates → bertopic_keywords_weights_with_dates.csv")
