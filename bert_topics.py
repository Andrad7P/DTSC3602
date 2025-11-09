import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


def main():
    # 1. Load the articles that your keyword_flagger created
    df = pd.read_csv("outseer_articles.csv")

    # 2. Build a single text field per article (title + full_text)
    df["doc"] = (
        df["title"].fillna("").astype(str)
        + ". "
        + df["full_text"].fillna("").astype(str)
    )

    docs = df["doc"].tolist()

    # 3. Create the embedding model (small, fast sentence-BERT)
    print("Loading sentence transformer model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 4. Create and fit the BERTopic model
    print("Fitting BERTopic model on documents...")
    topic_model = BERTopic(embedding_model=embed_model, verbose=True)
    topics, probs = topic_model.fit_transform(docs)

    # 5. Attach dominant topic back to each article
    df["topic"] = topics
    df.to_csv("outseer_articles_with_topics.csv", index=False)
    print("Saved article-level topics → outseer_articles_with_topics.csv")

    # 6. Overview of topics (size, name, etc.)
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv("bertopic_topics_overview.csv", index=False)
    print("Saved topic overview → bertopic_topics_overview.csv")

    # 7. Extract keywords + weights for each topic
    #    This is the “exact keywords + weight” part you own
    keyword_rows = []
    for topic_id in topic_info["Topic"]:
        # -1 is usually the "outlier" topic; skip if you don’t want it
        if topic_id == -1:
            continue

        # list of (word, weight) pairs
        words = topic_model.get_topic(topic_id)
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
    kw_df.to_csv("bertopic_keywords_weights.csv", index=False)
    print("Saved keyword weights → bertopic_keywords_weights.csv")


if __name__ == "__main__":
    main()
