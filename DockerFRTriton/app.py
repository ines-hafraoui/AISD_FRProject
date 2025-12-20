import logging
import os
import uvicorn
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile

# Import from pipeline (which handles the TritonService logic)
from pipeline import calculate_face_similarity, run_inference
from triton_service import (
    TRITON_HTTP_PORT,
    create_triton_client,
    stop_triton_server,
)

MODEL_REPO = Path(__file__).parent / "model_repository"

app = FastAPI(
    title="FR Triton API",
    description="Minimal FastAPI wrapper around Triton Inference Server.",
    version="0.1.0",
)

_server_handle: Optional[Any] = None
_triton_client: Optional[Any] = None
logger = logging.getLogger("fr_triton_app")

@app.on_event("startup")
def startup_event() -> None:
    global _triton_client
    # Since start.sh handles the download and server launch,
    # we just connect the client here.
    try:
        # Connect to the local Triton instance
        _triton_client = create_triton_client(f"localhost:{TRITON_HTTP_PORT}")
        logger.info("Successfully connected to Triton Client.")
    except Exception as exc:
        logger.error("Failed to connect to Triton: %s", exc)

@app.on_event("shutdown")
def shutdown_event() -> None:
    global _server_handle
    if _server_handle:
        stop_triton_server(_server_handle)
    _server_handle = None

@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/embedding", tags=["Face Recognition"])
async def embedding(image: UploadFile = File(..., description="Face image to embed")) -> dict[str, Any]:
    if _triton_client is None:
        raise HTTPException(status_code=503, detail="Triton client is not initialized.")

    content = await image.read()
    try:
        # This calls the wrapper in pipeline.py
        embedding_arr = run_inference(_triton_client, content)
        return {"embedding": embedding_arr.tolist()}
    except Exception as exc:
        logger.error(f"Inference error: {exc}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

@app.post("/face-similarity", tags=["Face Recognition"])
async def face_similarity(
    image_a: UploadFile = File(..., description="First face image"),
    image_b: UploadFile = File(..., description="Second face image"),
) -> dict[str, Any]:
    if _triton_client is None:
        raise HTTPException(status_code=503, detail="Triton client is not initialized.")

    content_a, content_b = await image_a.read(), await image_b.read()
    try:
        score = calculate_face_similarity(_triton_client, content_a, content_b)
        return {"similarity": float(score)}
    except Exception as exc:
        logger.error(f"Similarity error: {exc}")
        raise HTTPException(status_code=500, detail=f"Similarity failed: {exc}")

if __name__ == "__main__":
    # Match the port to what your start.sh or Dockerfile expects
    # If using your start.sh logic, this should be 3000
    uvicorn.run("app:app", host="0.0.0.0", port=3000, reload=False)