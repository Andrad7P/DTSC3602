import pandas as pd
from nltk.stem import PorterStemmer
import re


def merge_keyword_dates():
    # Load keyword weights & articles
    kw = pd.read_csv("bertopic_keywords_weights.csv")
    articles = pd.read_csv("outseer_articles.csv")

    # If no date column, warn and skip date extraction
    if "date" not in articles.columns:
        print("⚠️ WARNING: No 'date' column found in outseer_articles.csv. Assigning blank dates.")
        articles["date"] = ""

    else:
        # Convert date to datetime
        articles["date"] = pd.to_datetime(articles["date"], errors="coerce")

    # Build a lowercase text field
    articles["full_text_clean"] = (
        articles["title"].fillna("").astype(str) + " " +
        articles["full_text"].fillna("").astype(str)
    ).str.lower()

    # Stemmer for matching words
    stemmer = PorterStemmer()

    # Add stem columns
    kw["stem"] = kw["keyword"].apply(lambda x: stemmer.stem(str(x).lower()))
    articles["stem_text"] = articles["full_text_clean"].apply(
        lambda text: " ".join(stemmer.stem(w) for w in re.findall(r"\b\w+\b", text))
    )

    # Collect results
    results = []

    for _, row in kw.iterrows():
        stem = row["stem"]

        # Find articles where the stem word appears
        mask = articles["stem_text"].str.contains(rf"\b{stem}\b", regex=True)
        matched = articles[mask]

        if matched.empty:
            first_date = ""
            last_date = ""
            count = 0
        else:
            first_date = matched["date"].min()
            last_date = matched["date"].max()
            count = len(matched)

        results.append({
            "topic": row["topic"],
            "rank": row["rank"],
            "keyword": row["keyword"],
            "weight": row["weight"],
            "first_date": first_date,
            "last_date": last_date,
            "doc_count": count
        })

    # Output final file
    out = pd.DataFrame(results)
    out.to_csv("bertopic_keywords_with_dates.csv", index=False)
    print("✅ Saved → bertopic_keywords_with_dates.csv")


if __name__ == "__main__":
    merge_keyword_dates()
