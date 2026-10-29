"""
Generation module for the RAG pipeline. Combines user queries
with retrieved review context and prompts an LLM to produce
grounded, fact-checked responses based strictly on the reviews.
"""

import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from groq import Groq

# support running as module, or stand-alone script
try:
    from src.retriever import retrieve_reviews, format_context_for_llm
except ModuleNotFoundError:
    from retriever import retrieve_reviews, format_context_for_llm

load_dotenv()

CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))
FILTERING_MODEL = "openai/gpt-oss-20b" # lightweight model
MODEL_NAME = "openai/gpt-oss-120b"

FILTERING_PROMPT = """
You are a query analysis engine for a video game review search system. Given a raw user query, your task is to extract the underlying semantic search phrase and deduce any implied rating filters.

Instructions:
1. "clean_query": The core subject matter to search semantically. Remove filter words like "1 star", "worst", "best" to keep the semantic keywords focused (for example: "worst lag complaints about Skyrim" becomes something like "Skyrim lag delay")
2. "min_rating": If the query implies a minimum rating (e.g., "best", "top rated", "5 stars"), set this to the corresponding float value (1.0 to 5.0). If no minimum rating is implied, set this to null.
3. "max_rating": If the query implies a maximum rating (e.g., "worst", "1 star", "lowest rated"), set this to the corresponding float value (1.0 to 5.0). If no maximum rating is implied, set this to null.

Return ONLY a valid JSON object as follows:
{
  "clean_query": string,
  "min_rating": float or null,
  "max_rating": float or null
}
Do not include any other text or explanation.
"""

SYSTEM_PROMPT = """
You are an expert video game assistant answering user questions based strictly on real customer reviews.

Instructions:
1. Grounding: Answer the user's question using ONLY the provided reviews below. Do not assume facts or make up details that are not in the context.
2. Insufficient Context: If the context does not give you enough information to answer the question, clearly state that the reviews do not provide sufficient information.
3. Citations: When referencing a specific review, cite it directly (for example: [Review #1] or by referencing the specific game/rating).
4. Tone: Be informative, concise and objective.
"""

def extract_rating_intent(raw_query: str) -> Dict[str, Any]:
    """
    Decomposes natural language queries into a focused semantic
    search string with optional numerical rating boundaries.
    """
    try:
        response = CLIENT.chat.completions.create(
            model=FILTERING_MODEL,
            messages=[
                {"role": "system", "content": FILTERING_PROMPT},
                {"role": "user", "content": f"User Query: {raw_query}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(response.choices[0].message.content) if response.choices else {}
        clean = data.get("clean_query")
        return {
            "clean_query": clean.strip() if clean and clean.strip() else raw_query,
            "min_rating": data.get("min_rating"),
            "max_rating": data.get("max_rating"),
        }

    except Exception as e:
        # fallback to original query
        return {
            "clean_query": raw_query,
            "min_rating": None,
            "max_rating": None,
        }

def generate_answer(
    query: str,
    top_k: int = 4,
    game_filter: Optional[str] = None,
    min_rating: Optional[float] = None,
    max_rating: Optional[float] = None,
    auto_filter: bool = True,
) -> Dict[str, Any]:
    """
    Retrieves context for the user query, formats a prompt
    and then calls the LLm to generate a grounded response."""
    search_query = query
    applied_min_rating = min_rating
    applied_max_rating = max_rating

    # auto extract bounds
    if auto_filter and applied_min_rating is None and applied_max_rating is None:
        parsed = extract_rating_intent(query)
        search_query = parsed["clean_query"]
        applied_min_rating = parsed["min_rating"]
        applied_max_rating = parsed["max_rating"]

    reviews = retrieve_reviews(
        query=search_query,
        top_k=top_k,
        game_filter=game_filter,
        min_rating=applied_min_rating,
        max_rating=applied_max_rating,
    )

    if not reviews:
        return {
            "query": query,
            "search_query_used": search_query,
            "filters_used": {
                "game_filter": game_filter,
                "min_rating": applied_min_rating,
                "max_rating": applied_max_rating
            },
            "answer": "No relevant reviews found for the given query and filters.",
            "retrieved_reviews": [],
        }

    context = format_context_for_llm(reviews)

    prompt_user_message = f"""User Question: {query}

Retrieved Reviews:
{context}

Please answer the user's question based strictly on the above reviews, following the rules provided in the system prompt."""

    response = CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_user_message},
        ],
        temperature=0.2,
    )   

    answer = response.choices[0].message.content if response.choices else "No answer generated."

    return {
        "query": query,
        "search_query_used": search_query,
        "filters_used": {
            "game_filter": game_filter,
            "min_rating": applied_min_rating,
            "max_rating": applied_max_rating
        },
        "answer": answer,
        "retrieved_reviews": reviews,
    }

if __name__ == "__main__":
    # testing
    test_question = "What do people say about Skyrim VR?"
    print(f"Testing RAG generation for question: '{test_question}'\n")

    result = generate_answer(
        query=test_question,
        top_k=10
    )

    print("--- Search Parameters Used ---")
    print(f"Clean Query: {result.get('search_query_used')}")
    print(f"Filters: {result.get('filters_used')}")

    print("\n--- Answer from the LLM ---")
    print(result["answer"])
