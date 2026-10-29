"""
FastAPI app for the RAG system. Provides endpoints for
health check, semantic search and RAG context assembly.
"""

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# support running as module, or stand-alone script
try:
    from src.retriever import retrieve_reviews, format_context_for_llm
except ModuleNotFoundError:
    from retriever import retrieve_reviews, format_context_for_llm

app = FastAPI(
    title="Game Review RAG API",
    description="Semantic search and context retrieval for video game reviews...",
)

# --- schemas ---
class ReviewResult(BaseModel):
    """Schema representing an individual review result"""
    review_text: str = Field(..., description="The text of the retrieved review")
    metadata: Dict[str, Any] = Field(..., description="Metadata associated with the review")
    distance: Optional[float] = Field(None, description="Cosine distance from query vector")

class SearchResponse(BaseModel):
    """API response schema for raw semantic search results"""
    query: str = Field(..., description="The original search query")
    count: int = Field(..., description="The number of reviews retrieved matching the query and filters")
    results: List[ReviewResult] = Field(..., description="List of retrieved reviews matching the query and filters")

class ContextResponse(BaseModel):
    """API response schema with context ready for LLM prompting"""
    query: str = Field(..., description="The original search query")
    count: int = Field(..., description="The number of reviews retrieved matching the query and filters")
    context: str = Field(..., description="Retrieved reviews formatted into context strings ready for LLM generation")


# --- endpoints ---
@app.get("/health", summary="Health check endpoint")
def health_check():
    """Simple health check endpoint to verify the API is running"""
    return {"status": "ok"}

@app.get("/reviews/search", response_model=SearchResponse, summary="Semantic search for game reviews")
def search_reviews_endpoint(
    query: str = Query(..., description="The search query string"),
    top_k: int = Query(4, description="Number of results to retrieve"),
    game_filter: Optional[str] = Query(None, description="Filter results by specific game title"),
    min_rating: Optional[float] = Query(None, ge=1.0, le=5.0, description="Filter results by minimum rating"),
    max_rating: Optional[float] = Query(None, ge=1.0, le=5.0, description="Filter results by maximum rating"),
):
    """
    Gets the most relevant reviews matching a query string with optional metadata filtering.
    """
    try:
        retrieved_reviews = retrieve_reviews(
            query=query,
            top_k=top_k,
            game_filter=game_filter,
            min_rating=min_rating,
            max_rating=max_rating
        )
        return SearchResponse(
            query=query,
            count=len(retrieved_reviews),
            results=retrieved_reviews
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reviews/context", response_model=ContextResponse, summary="Retrieve context for LLM generation")
def get_rag_context_endpoint(
    query: str = Query(..., description="The search query string"),
    top_k: int = Query(4, description="Number of results to retrieve"),
    game_filter: Optional[str] = Query(None, description="Filter results by specific game title"),
    min_rating: Optional[float] = Query(None, ge=1.0, le=5.0, description="Filter results by minimum rating"),
    max_rating: Optional[float] = Query(None, ge=1.0, le=5.0, description="Filter results by maximum rating"),
):
    """
    Gets reviews and formats them into an assembled prompt string ready for an LLM.
    """
    try:
        retrieved_reviews = retrieve_reviews(
            query=query,
            top_k=top_k,
            game_filter=game_filter,
            min_rating=min_rating,
            max_rating=max_rating
        )
        formatted_context_str = format_context_for_llm(retrieved_reviews)
        return ContextResponse(
            query=query,
            count=len(retrieved_reviews),
            context=formatted_context_str
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))