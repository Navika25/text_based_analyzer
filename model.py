import random
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class HybridSentimentModel:
    def __init__(self, use_mock=True):
        self.use_mock = use_mock
        self.vader = SentimentIntensityAnalyzer()

    def preprocess_text(self, text):
        return text.lower().strip()

    def get_vader_sentiment(self, text):
        score = self.vader.polarity_scores(text)
        return score['compound']

    def get_textblob_sentiment(self, text):
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def mock_predict(self, text):
        vader_score = self.get_vader_sentiment(text)
        blob_score = self.get_textblob_sentiment(text)

        # Combine both scores
        final_score = (vader_score + blob_score) / 2

        # Classification
        if final_score > 0.2:
            sentiment = "Positive"
        elif final_score < -0.2:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return {
            "sentiment": sentiment,
            "score": round(final_score, 3)
        }

    def predict(self, text):
        text = self.preprocess_text(text)

        if self.use_mock:
            return self.mock_predict(text)
        else:
            return self.mock_predict(text)  # fallback
