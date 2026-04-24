from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
from model import HybridSentimentModel

app = FastAPI(title="Hybrid Sentiment Analysis API")

# Initialize model
sentiment_model = HybridSentimentModel(use_mock=True)

class SentimentRequest(BaseModel):
    text: str

@app.post("/api/analyze")
async def analyze_sentiment(request: SentimentRequest):
    result = sentiment_model.predict(request.text)
    score = result["score"]

    # Convert score to probabilities (for UI compatibility)
    if score > 0.2:
        probs = [0.05, 0.1, 0.85]
    elif score < -0.2:
        probs = [0.85, 0.1, 0.05]
    else:
        probs = [0.2, 0.6, 0.2]

    labels = ["Negative", "Neutral", "Positive"]
    max_idx = probs.index(max(probs))

    return {
        "text": request.text,
        "sentiment": labels[max_idx],
        "confidence": round(probs[max_idx] * 100, 2),
        "probabilities": probs   # IMPORTANT for frontend
    }

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

# Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
