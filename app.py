from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
from model import HybridSentimentModel

app = FastAPI(title="Hybrid Sentiment Analysis API")

# Initialize model (mock mode ON → no tensorflow needed)
sentiment_model = HybridSentimentModel(use_mock=True)

class SentimentRequest(BaseModel):
    text: str

@app.post("/api/analyze")
async def analyze_sentiment(request: SentimentRequest):
    probs = sentiment_model.predict(request.text)

    labels = ["Negative", "Neutral", "Positive"]
    max_idx = probs.index(max(probs))

    return {
        "text": request.text,
        "sentiment": labels[max_idx],
        "confidence": round(probs[max_idx] * 100, 2),
        "probabilities": {
            "Negative": round(probs[0] * 100, 2),
            "Neutral": round(probs[1] * 100, 2),
            "Positive": round(probs[2] * 100, 2)
        },
        "model_components": {
            "ml_layer": "SVM + Random Forest + Gradient Boosting",
            "dl_layer": "Mock DL (No TensorFlow)",
            "ml_contribution": 50,
            "dl_contribution": 50
        }
    }

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

# Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Local run (for your laptop)
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
