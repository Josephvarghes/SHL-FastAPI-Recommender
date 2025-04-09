from fastapi import FastAPI
from pydantic import BaseModel
from app.embedder  import prepare_embeddings, search_assessments
import json

# Load assessment data
with open("data/assessments.json", "r") as f:
    assessments = json.load(f)

assessments = prepare_embeddings(assessments)  # Embed once on startup

# FastAPI instance
app = FastAPI(title="SHL Assessment Recommender API")

# Request format
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# POST endpoint
@app.post("/recommend")
def recommend_assessments(request: QueryRequest):
    results = search_assessments(request.query, assessments, request.top_k)
    return {"query": request.query, "matches": results}
