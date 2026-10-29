"""
Queries ChromaDB with optional metadata filters and formats results
into context strings ready for LLM generation. 
"""

from typing import Any, Dict, List, Optional
from src.vector_store import get_or_create_collection

def retrieve_reviews(
        query: str,
        top_k: int = 4,
        game_filter: Optional[str] = None,
        min_rating: Optional[float] = None,
        max_rating: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Queries ChromaDB for most relevant reviews matching a query string
    with support for metadata filtering.
    """

    collection = get_or_create_collection()

    filter_conditions = []

    if game_filter:
        filter_conditions.append({"product_name": {"$eq": game_filter}})
    if min_rating is not None:
        filter_conditions.append({"rating": {"$gte": float(min_rating)}})
    if max_rating is not None:
        filter_conditions.append({"rating": {"$lte": float(max_rating)}})

    # combine filters together for ChromaDB
    where_clause = None

    if len(filter_conditions) == 1:
        where_clause = filter_conditions[0]
    elif len(filter_conditions) > 1:
        where_clause = {"$and": filter_conditions}

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where_clause
    )

    retrieved_reviews = []
    if results and "documents" in results and len(results["documents"]) > 0:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results else [None] * len(docs)

        for doc, meta, dist in zip(docs, metas, distances):
            retrieved_reviews.append({
                "review_text": doc,
                "metadata": meta,
                "distance": dist
            })

    return retrieved_reviews

def format_context_for_llm(retrieved_reviews: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved reviews into a structured text prompt
    ready to be fed into an LLM for further processing or generation.
    """

    if not retrieved_reviews:
        return "No relevant reviews found."

    context_blocks = []

    for index, item in enumerate(retrieved_reviews, 1):
        metadata = item['metadata']
        product_name = metadata.get("product_name", "Unknown Product")
        rating = metadata.get("rating", "N/A")
        sentiment_score = metadata.get("sentiment_score", "N/A")
        review_text = item.get("review_text", "")

        context_block = (
            f"Review #{index}\n"
            f"Product: {product_name}\n"
            f"Rating: {rating}/5.0 | Sentiment Score: {sentiment_score}\n"
            f"Review Text: {review_text}\n"
        )

        context_blocks.append(context_block)

    return "\n---\n".join(context_blocks)

if __name__ == "__main__": # test retrieval
    # test a plain query first
    test_query = "laggy and poor performance"
    print(f"--- Testing Plain Query with No Filters (Query = '{test_query}' ---")
    print(f"Retrieving reviews for query: '{test_query}'")

    results = retrieve_reviews(query=test_query, top_k=3)
    formatted_context = format_context_for_llm(results)

    print("-- Formatted Context for LLM --")
    print(formatted_context)

    # test a query with filters
    print(f"--- Testing Plain Query with Filters (Query = '{test_query}, Rating <= 2.0) ---")
    filtered_results = retrieve_reviews(
        query=test_query,
        top_k=3,
        max_rating=2.0
    )
    filtered_formatted_context = format_context_for_llm(filtered_results)
    print(filtered_formatted_context)