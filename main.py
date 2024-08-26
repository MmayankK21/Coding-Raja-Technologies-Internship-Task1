import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pickle
from tqdm import tqdm
import scipy.sparse as sp

dataset_path = "C:/Users/MAYANK/Desktop/twitter_training.csv"
data = pd.read_csv(dataset_path)

data.columns = ['id', 'user', 'sentiment', 'text']

sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
data['sentiment'] = data['sentiment'].map(sentiment_map)

def preprocess_text(text):
    if not isinstance(text, str):
        text = ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text

data['text'] = data['text'].apply(preprocess_text)

data = data.dropna(subset=['sentiment'])

train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['text'], data['sentiment'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=10000)
train_features = vectorizer.fit_transform(train_texts)
test_features = vectorizer.transform(test_texts)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

train_labels = torch.tensor(train_labels.values, dtype=torch.long)
test_labels = torch.tensor(test_labels.values, dtype=torch.long)

class SparseDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx].toarray().squeeze(), dtype=torch.float32), self.labels[idx]

# Create DataLoader
batch_size = 64
train_dataset = SparseDataset(train_features, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = SparseDataset(test_features, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model
from model import SentimentModel

input_dim = train_features.shape[1]
model = SentimentModel(input_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    torch.save(model.state_dict(), 'sentiment_model.pth')

train_model(model, train_loader, criterion, optimizer, num_epochs=10)

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

evaluate_model(model, test_loader)
