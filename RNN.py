import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# 🟢 Step 1: Load and Preprocess Data
df = pd.read_csv("hotel_reviews.csv")

# Combine positive and negative reviews
df['review_text'] = df['Positive_Review'].fillna('') + " " + df['Negative_Review'].fillna('')
df['sentiment'] = df.apply(lambda row: 1 if row['Review_Total_Positive_Word_Counts'] >= row['Review_Total_Negative_Word_Counts'] else 0, axis=1)

# Sample a smaller dataset for training speed
df_sampled = df.sample(n=5000, random_state=42)

# Tokenization
nltk.download('punkt_tab')
nltk.download("punkt")
df_sampled["tokens"] = df_sampled["review_text"].apply(word_tokenize)

# Create vocabulary
all_words = [word for tokens in df_sampled["tokens"] for word in tokens]
word_freq = Counter(all_words)
vocab = {word: i + 2 for i, (word, _) in enumerate(word_freq.most_common(10000))}  # Most frequent 10,000 words
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

# Convert words to indices
def encode_text(tokens):
    return [vocab.get(word, 1) for word in tokens]  # 1 is <UNK>

df_sampled["encoded"] = df_sampled["tokens"].apply(encode_text)

# Pad sequences
max_len = 200  # Limit sequence length
def pad_sequence(seq, max_len):
    return seq[:max_len] + [0] * (max_len - len(seq))  # Pad with 0s

df_sampled["padded"] = df_sampled["encoded"].apply(lambda x: pad_sequence(x, max_len))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df_sampled["padded"].tolist(), df_sampled["sentiment"].tolist(), test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train, X_test = torch.tensor(X_train, dtype=torch.long), torch.tensor(X_test, dtype=torch.long)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# Create a Dataset class
class ReviewDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoader
batch_size = 32
train_dataset = ReviewDataset(X_train, y_train)
test_dataset = ReviewDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 🟢 Step 2: Define LSTM Model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, output_dim=1, n_layers=2, dropout=0.5):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        final_output = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(final_output)

# Initialize model
vocab_size = len(vocab)
model = SentimentLSTM(vocab_size)

# Move model to MPS (Apple Silicon) or CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# 🟢 Step 3: Train the Model
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, criterion, optimizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

train(model, train_loader, criterion, optimizer, device, epochs=3)

# 🟢 Step 4: Evaluate the Model
def evaluate(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).squeeze()
            preds = (preds > 0.5).float()  # Convert probabilities to binary labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Test Accuracy: {acc:.4f}")

evaluate(model, test_loader, device)
