"""
Since the full dataset is 2GB+ in size, we are going to create a smaller sample dataset to work with.
This script will find the 15 most popular video games and collect up to 50 reviews for that game.
We will then grade how positive or negative each review is (from -1 to 1) and save the results to a CSV file (`data/video_games_demo.csv`)
"""

from collections import defaultdict
import os
import pandas as pd
import nltk 
from nltk.sentiment import SentimentIntensityAnalyzer
from datasets import load_dataset

# download the VADER lexicon (a list of words with positive/negative weights)
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

HF_REVIEWS_URL = "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/Video_Games.jsonl"
HF_META_URL = "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/meta_categories/meta_Video_Games.jsonl"

def ingest_video_game_reviews(num_of_products=15, max_reviews_per_product=50):
    """
    Streams a subset of Video Game reviews from the Hugging Face dataset,
    computes sentiment scores, and saves to data/video_games_demo.csv
    """

    # 1) choosing 15 games
    print("1/3: Finding the 15 most popular games from the dataset...")

    review_stream = load_dataset(
        "json", 
        data_files={"train": HF_REVIEWS_URL},
        split="train",
        streaming=True,
    )

    asin_counts = defaultdict(int)
    reviews_buffer = defaultdict(list)

    for i, review in enumerate(review_stream):
        if i >= 50_000:
            break

        asin = review.get("parent_asin")
        text = review.get("text", "")
        title = review.get("title", "")

        if not asin or not text or len(text.strip()) < 10:
            continue

        asin_counts[asin] += 1

        if len(reviews_buffer[asin]) < max_reviews_per_product:
            reviews_buffer[asin].append(review)

        if (i + 1) % 10_000 == 0:
            print(f"  Scanned {i + 1:,} reviews... ({len(asin_counts)} unique ASINs seen)")

    sorted_asins = sorted(asin_counts.items(), key=lambda item: item[1], reverse=True)
    target_asins = {asin for asin, _ in sorted_asins[:num_of_products]}
    print(f"Identified top {len(target_asins)} games...")

    # 2) fetching product titles from metadata
    print(f"2/3: Fetching product titles from metadata...")

    import json
    import urllib.request

    meta_url = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Video_Games.jsonl"
    req = urllib.request.Request(meta_url, headers={"User-Agent": "Mozilla/5.0"})

    # meta_stream = load_dataset(
    #     "json",
    #     data_files={"train": HF_META_URL},
    #     split="train",
    #     streaming=True,
    # )

    product_titles = {}
    with urllib.request.urlopen(req) as response:
        for line in response:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line.decode("utf-8"))
            asin = item.get("parent_asin")

            if asin in target_asins and asin not in product_titles:
                raw_title = item.get("title", "").strip()
                product_titles[asin] = raw_title if raw_title else f"Game_{asin}"

                if len(product_titles) == len(target_asins):
                    break

    # product_titles = {}
    # for item in meta_stream:
    #     asin = item.get("parent_asin")
    #     if asin in target_asins and asin not in product_titles:
    #         raw_title = item.get("title", "").strip()
    #         product_titles[asin] = raw_title if raw_title else f"Game_{asin}"

    #         if len(product_titles) == len(target_asins):
    #             break

    # 3) running VADER sentiment analysis and building dataset
    print("3/3: Running VADER sentiment analysis and building dataset...")

    collected_reviews = []

    for asin in target_asins:
        game_title = product_titles.get(asin, f"Game_{asin}")
        for review in reviews_buffer[asin]:
            clean_title = review.get("title", "").strip()
            clean_text = review.get("text", "").strip()
            full_text = f"{clean_title} || {clean_text}" if clean_title else clean_text

            sentiment_score = sia.polarity_scores(full_text)["compound"]

            collected_reviews.append({
                "product_id": asin,
                "product_name": game_title,
                "rating": float(review.get("rating", 0)),
                "sentiment_score": round(sentiment_score, 3),
                "review_title": clean_title,
                "review_text": clean_text,
                "verified": bool(review.get("verified_purchase", False)),
            })

    df = pd.DataFrame(collected_reviews)
    os.makedirs("data", exist_ok=True)
    output_path = "data/video_games_demo.csv"
    df.to_csv(output_path, index=False)

    print(
        f"Done! Saved {len(df)} scored reviews across "
        f"{df['product_id'].nunique()} games to {output_path}"
    )

if __name__ == "__main__":
    ingest_video_game_reviews()