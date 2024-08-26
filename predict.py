import torch
from model import SentimentModel
import pickle
import re

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text

def predict_sentiment(text, model, vectorizer):
    model.eval()
    preprocessed_text = preprocess_text(text)
    features = vectorizer.transform([preprocessed_text])
    features = torch.tensor(features.toarray(), dtype=torch.float32)

    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        sentiments = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        return sentiments[predicted.item()]

if __name__ == "__main__":
    input_dim = 10000
    model = SentimentModel(input_dim)
    model.load_state_dict(torch.load('sentiment_model.pth'))

    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Predict sentiment
    while True:
        text = input("Enter text to predict sentiment (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break
        sentiment = predict_sentiment(text, model, vectorizer)
        print(f"Sentiment: {sentiment}")