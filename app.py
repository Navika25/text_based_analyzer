from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
from model import HybridSentimentModel

app = FastAPI(title="Hybrid Sentiment Analysis API")

# Initialize the model 
# (use_mock=True enables running out-of-the-box without waiting hours for DL weights)
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
            "dl_layer": "GloVe + Bi-LSTM + Self-Attention",
            "ml_contribution": 50,
            "dl_contribution": 50
        }
    }

# Make sure static dir exists
os.makedirs("static", exist_ok=True)

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
