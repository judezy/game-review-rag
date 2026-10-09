"""
Loads the video game reviews data, generates semantic text embeddings,
and indexes review text alongside metadata (sentiment score, rating, product)
in ChromaDB to enable semantic search and context retrieval for RAG.
"""

import os
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd

CSV_PATH = "data/video_games_demo.csv"
DB_DIR = "chroma_db"
COLLECTION_NAME = "game_reviews"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # from sentence-transformers

def get_embedding_function():
    """Returns a ChromaDB embedding function using the specified sentence-transformers model."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

def get_or_create_collection(client=None):
    """Retrieves the collection if it exists, otherwise creates a new one."""
    if client is None:
        client = chromadb.PersistentClient(path=DB_DIR)

    embedding_function = get_embedding_function()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata = {"hnsw:space": "cosine"}
    )
    
    return collection

def populate_vector_store(force_reload: bool = False):
    """Reads the CSV data and inserts formatted reviews with metadata into the ChromaDB collection."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found at {CSV_PATH}. Please ensure the file exists or run src/ingest.py first.")

    client = chromadb.PersistentClient(path=DB_DIR)

    if force_reload:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}' due to force_reload=True.")
        except Exception:
            pass

    collection = get_or_create_collection(client)

    current_count = collection.count()
    if current_count > 0:
        print(f"Collection '{COLLECTION_NAME}' already has {current_count} reviews, so skipping ingestion. Use force_reload=True to overwrite.")
        return
    
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}.")

    documents = []
    metadatas = []
    ids = []

    for index, row in df.iterrows():
        review_title = str(row.get("review_title", "")).strip()
        review_text = str(row.get("review_text", "")).strip()
        game_name = str(row.get("product_name", "")).strip()

        if review_title and review_title.lower() != "nan":
            full_text = f"Game: {game_name} | Review: {review_title} - {review_text}"
        else:
            full_text = f"Game: {game_name} | Review: {review_text}"

        metadata = {
            "product_id": str(row.get("product_id", "")),
            "product_name": game_name,
            "rating": float(row.get("rating") if pd.notna(row.get("rating")) else 0.0),
            "sentiment_score": float(row.get("sentiment_score") if pd.notna(row.get("sentiment_score")) else 0.0),
            "verified": bool(row.get("verified", False)),
        }

        documents.append(full_text)
        metadatas.append(metadata)
        ids.append(f"review_{index}")

    # insert to ChromaDB in batches
    batch_size = 100
    print(f"Embedding and inserting {len(documents)} reviews into ChromaDB in batches of {batch_size}...")

    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
        print(f"Inserted reviews {i + 1} to {end} into ChromaDB.")

    print(f"Finished! Total reviews in vector store: {collection.count()}")

def test_query(query_text: str, n_results: int = 3):
    """Runs a semantic search against the collection to test retrieval."""
    print(f"\n--- Testing Semantic Search: '{query_text}' ---")
    
    collection = get_or_create_collection()

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )

    for i in range(n_results):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        distance = results['distances'][0][i] if "distances" in results else None
        dist_str = f"{distance:.4f}" if distance is not None else "N/A"

        print(f"\nResult #{i + 1} (Cosine Distance: {dist_str}):")
        print(f"    Game: {meta.get('product_name', 'N/A')}")
        print(f"    Rating: {meta.get('rating', 'N/A')} / 5.0 | Sentiment: {meta.get('sentiment_score', 'N/A')}")
        print(f"    Review: {doc[:200]}") # limit to first 200 chars for terminal output


if __name__ == "__main__":
    populate_vector_store()
    test_query("crashing and unplayable lag")