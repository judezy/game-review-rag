"""
Streamlit UI for the RAG system. Communicates with the FastAPI backend to
provide interactive Q&A, filter inspection and retrieved review browsing.
"""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Game Review RAG System",
    layout="wide"
)

st.title("Game Review RAG System")
st.caption("System powered by ChromaDB, Groq API, and FastAPI.")

# sidebar
with st.sidebar:
    st.header("Search and Retrieval Settings")
    top_k = st.slider("Number of reviews to retrieve", min_value=1, max_value=10, value=4, step=1)
    auto_filter = st.checkbox("Auto-extract rating filters from query", value=True)

    st.markdown("---")
    st.subheader("Manual Overrides (Optional)")
    game_filter = st.text_input("Exact Game Title", value="", placeholder="e.g. Skyrim VR - PlayStation 4")
    min_rating = st.number_input("Minimum Rating", min_value=1.0, max_value=5.0, value=None, step=0.5)
    max_rating = st.number_input("Maximum Rating", min_value=1.0, max_value=5.0, value=None, step=0.5)

    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        if health.status_code == 200:
            st.success("API Status: Online")
        else:
            st.error(f"API Status: Error")
    except requests.exceptions.RequestException:
        st.error("API Status: Offline (Run FastAPI first)")

# main query
query = st.text_input(
    "Ask a question about games or gear:",
    placeholder="eg. What do 1-star reviews complain about for controllers?"
)

if st.button("Search and Answer", type="primary") and query.strip():
    payload = {
        "query": query,
        "top_k": top_k,
        "game_filter": game_filter if game_filter.strip() else None,
        "min_rating": min_rating if min_rating is not None else None,
        "max_rating": max_rating if max_rating is not None else None,
        "auto_filter": auto_filter
    }

    with st.spinner("Analysing query..."):
        try:
            response = requests.post(f"{API_URL}/rag/ask", json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()

                st.markdown("### Search Analysis")

                # search info
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Cleaned Query", data.get("search_query_used", query))
                with col2:
                    st.metric("Min Rating Filter", str(data["filters_used"].get("min_rating")))
                with col3:
                    st.metric("Max Rating Filter", str(data["filters_used"].get("max_rating")))
                with col4:
                    st.metric("Reviews Retrieved", len(data.get("retrieved_reviews", [])))

                st.markdown("---")

                # answer
                st.markdown("### Answer")
                st.markdown(data['answer'])

                # source 
                st.markdown("---")
                st.markdown("### Retrieved Context Reviews")

                reviews = data.get("retrieved_reviews", [])

                if not reviews:
                    st.info("No context reviews returned.")
                else:
                    for i, r in enumerate(reviews, 1):
                        meta = r.get("metadata", {})
                        rating = meta.get("rating", "N/A")
                        dist = r.get("distance")
                        dist_str = f"{dist:.4f}" if dist is not None else "N/A"
                        product_name = meta.get("product_name", "Unknown Product")
                        sentiment_score = meta.get("sentiment_score", "N/A")
                        verified = "Verified Purchase" if meta.get("verified") else "Unverified"

                        with st.expander(f"Review #{i} - {product_name} (Rating: {rating}/5 | Distance: {dist_str})"):
                            st.write(r["review_text"])
                            st.caption(f"Status: {verified} | Sentiment Score: {sentiment_score}")

            else:
                st.error(f"Error: {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to communicate with FastAPI backend: {e}")