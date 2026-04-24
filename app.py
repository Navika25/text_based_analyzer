@app.post("/api/analyze")
async def analyze_sentiment(request: SentimentRequest):
    result = sentiment_model.predict(request.text)

    score = result["score"]

    # Convert score into pseudo probabilities
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
        "probabilities": {
            "Negative": round(probs[0] * 100, 2),
            "Neutral": round(probs[1] * 100, 2),
            "Positive": round(probs[2] * 100, 2)
        },
        "model_components": {
            "ml_layer": "VADER + TextBlob",
            "dl_layer": "Simulated",
            "note": "Lightweight hybrid model"
        }
    }
