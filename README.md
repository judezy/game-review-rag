# Game Review RAG & Sentiment Analyzer

An AI-powered tool that lets shoppers search and analyze Amazon video game reviews using natural questions instead of keyword filters. 

It combines Retrieval-Augmented Generation (RAG) with sentiment analysis to summarize recurring feedback, highlight common complaints, and cite authentic customer experiences.

---

## Why this project?
Traditional review browsing requires manually scrolling through long reviews and relying on broad, aggregated ratings that often obscure specific hardware faults or software issues.

This system solves that by:
- **Understanding Intent:** Automatically extracting implicit rating boundaries from natural queries (eg: *"What do 1-star reviews complain about?"* $\rightarrow$ filter sets `rating <=1.0`)
- **Preventing Hallucination:** Answers are grounded strictly in the authentic customer reviews with citations.
- **Providing Visual Context:** Providing an interactive visual dashboard that displays sentiment scores, review distances and verified purchase flags on each review.

---

## Pipeline
```text
User Question (e.g., "What do 1-star reviews complain about for controllers?")
                               │
                               ▼
                [Intent Router: gpt-oss-20b]
          ├── Extracted Topic: "controller complaints"
          └── Extracted Filter: {"rating": {"$lte": 1.0}}
                               │
                               ▼
                   [Vector Store: ChromaDB]
          ├── Model: all-MiniLM-L6-v2 (Offline Mode)
          └── Metadata Filter: {"rating": {"$lte": 1.0}}
                               │
                               ▼
                 [Retrieved Reviews Context]
                               │
                               ▼
               [Synthesis Engine: gpt-oss-120b]
          └── Grounded, cited summary strictly from context
                               │
                               ▼
                 [Streamlit UI / FastAPI JSON]
```

---

## Key Features
- **Query Decomposition & Intent Routing:** Uses `openai/gpt-oss-20b` on Groq to isolate semantic keywords from quantitative constraints
- **Evidence Grounded Generation:** `openai/gpt-oss-120b` generates objective answers, flagging missing information and citing supporting reviews directly. 
- **Decoupled Architecture:** Clean seperation between vector ingestion, FastAPI REST endpoints and a simple Streamlit frontend.

---

## Repository Structure
```text
game-review-rag/
├── data/                  # Source review datasets (e.g video_games_demo.csv)
├── src/
│   ├── __init__.py        # Package initializer
│   ├── api.py             # FastAPI endpoints and Pydantic schemas
│   ├── generator.py       # Query intent router and LLM synthesis engine
│   ├── ingest.py          # Data preprocessing, cleaning, and batch embedding pipeline
│   ├── retriever.py       # Query vector retrieval and composite metadata filtering
│   └── vector_store.py    # ChromaDB client initialization and collection management
├── .gitignore
├── README.md
├── frontend.py            # Streamlit interactive UI
└── requirements.txt
```

---

## Quickstart
### 1. Setup
```bash
git clone https://github.com/judezy/game-review-rag.git
cd game-review-rag

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the project root and fill it like this:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Ingest Data (First Run Only)
Populate your local ChromaDB vector store using the demo dataset:

```bash
python3 -m src.ingest
```

### 4. Run
Launch the backend and frontend in seperate terminal tabs:

```bash
# Terminal 1: Backend
uvicorn src.api:app --reload

# Terminal 2: Frontend
streamlit run frontend.py
```

Then, you will have available:

- **Web UI:** `http://localhost:8501`
- **Swagger API Docs:** `http://127.0.0.1:8000/docs`

---

## API Reference
<details>
<summary><b>View Endpoints & Example Data</b></summary>

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | API status |
| `GET` | `/reviews/search` | Semantic search with optional metadata filters |
| `POST` | `/rag/ask` | Full RAG pipeline |

#### Example Request (`POST /rag/ask`)
```json
{
    "query": "What do the 1-star reviews complain about for controllers?",
    "top_k": 3,
    "auto_filter": true
}
```

#### Example Response
```json
{
    "query": "What do the 1-star reviews complain about for controllers?",
    "search_query_used": "controller complaints",
    "filters_used": {"min_rating": null, "max_rating": 1.0},
    "answer": "The 1-star reviews primarly focus on ....",
    "retrieved_reviews": [ ... ]
}
```

</details>

---

## Tech Stack
- **Frameworks:** FastAPI, Streamlit, Pydantic
- **Vector Database:** ChromaDB
- **Embeddings**: `all-MiniLM-L6-v2`
- **Generation:** Groq API (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`)