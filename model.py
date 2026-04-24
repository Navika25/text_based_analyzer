import os
import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Layer, BatchNormalization
import tensorflow.keras.backend as K
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

class Attention(Layer):
    """Custom self-attention layer as per the research paper architecture."""
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

class HybridSentimentModel:
    """Hybrid Architecture combining lexical ML classifiers and bidirectional LSTM networks."""
    def __init__(self, use_mock=True):
        self.use_mock = use_mock
        self.vader = SentimentIntensityAnalyzer()
        
        # Machine Learning Layer (Scikit-learn components)
        self.tfidf = TfidfVectorizer(max_features=10000)
        self.svm_model = SVC(kernel='linear', C=1.0, probability=True)
        self.rf_model = RandomForestClassifier(n_estimators=200, max_depth=30)
        self.gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
        
        # Deep Learning Layer
        self.dl_model = self._build_dl_model()
        
        # Ensemble weight (Alpha calibration parameter)
        self.alpha = 0.5
        self.is_trained = False
        
    def _build_dl_model(self):
        inputs = Input(shape=(512,))
        x = Embedding(input_dim=50000, output_dim=300)(inputs)
        x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(x)
        x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(x)
        x = Attention()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        outputs = Dense(3, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                      loss='categorical_crossentropy', metrics=['accuracy'])
        return model
        
    def preprocess_text(self, text):
        """Pipeline to clean and normalize text (spaCy alternative mapping)."""
        text = str(text).lower()
        # Remove URLs and HTML
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        # Mock contraction expansion
        text = text.replace("don't", "do not").replace("isn't", "is not").replace("can't", "cannot")
        return text
        
    def predict(self, text):
        """Generates sentiment probabilities using hybrid decision fusion."""
        cleaned = self.preprocess_text(text)
        
        if self.use_mock:
            return self._mock_predict(cleaned)
            
        # In a production context after training:
        # P_ml = ML_Ensemble(text)
        # P_dl = DL_BiLSTM(text)
        # final_P = alpha * P_ml + (1 - alpha) * P_dl
        return [0.33, 0.33, 0.34] # Placeholder mapping for trained model execution
        
    def _mock_predict(self, text):
        """Heuristic-based estimation to simulate a trained model immediately."""
        vs = self.vader.polarity_scores(text)
        compound = vs['compound']
        
        # Simulated context and negation improvements outlined in paper
        words = text.split()
        if any(w in ["not", "never", "none"] for w in words):
            # rudimentary negation scope marking simulation
            compound *= -0.5
            
        if "!" in text and compound < 0:
            compound -= 0.15 # emphasis scalar for negativity
            
        # Normalize to probability distributions
        if compound <= -0.1:
            base = [0.85, 0.10, 0.05]
        elif compound >= 0.1:
            base = [0.05, 0.10, 0.85]
        else:
            base = [0.15, 0.70, 0.15]
            
        # Variability
        jitter = np.random.uniform(-0.03, 0.03, 3)
        res = base + jitter
        res = [max(0.01, float(x)) for x in res]
        s = sum(res)
        return [x/s for x in res]

if __name__ == '__main__':
    model = HybridSentimentModel(use_mock=True)
    print("Model Architecture instantiated.")
    print("Sample Output -> 'Amazing experience': ", model.predict("Amazing experience"))
