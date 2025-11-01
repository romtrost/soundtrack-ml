# search_app.py - API facing
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from orchestrator import SearchOrchestrator
from utils.logging import setup_logger
from typing import Any

app = FastAPI()
orchestrator = SearchOrchestrator()

logger = setup_logger(__name__)

class SearchRequest(BaseModel):
    query: str
    #search_types: list[str] | None = None  # e.g., ["track", "album"]
    #top_k: int = 10

class SearchResponse(BaseModel):
    query: str
    results: Any # TODO: bad practice, fix up later
    #search_types_used: list[str]

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    logger.info(f"Received search request: {request}")
    try:
        results = orchestrator.search(
            query=request.query,
            #search_types=request.search_types, # TODO: this is actually imoportant, what if the user clicks on the track filter for example
            #top_k=request.top_k
        )
        return SearchResponse(
            query=request.query,
            results=results,
            #search_types_used=results.get("search_types", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)