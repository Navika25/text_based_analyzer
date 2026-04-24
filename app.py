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

    return {
        "text": request.text,
        "sentiment": result["sentiment"],
        "confidence": round(abs(result["score"]) * 100, 2),
        "score": result["score"],
        "model_components": {
            "ml_layer": "VADER + TextBlob",
            "dl_layer": "Removed (for deployment compatibility)",
            "note": "Lightweight hybrid model used"
        }
    }

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

# Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
