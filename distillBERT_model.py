import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#Last inn datasettet
df = pd.read_csv("hotel_reviews.csv")

# Kombiner positive og negative anmeldelser til én tekst
df['review_text'] = df['Positive_Review'].fillna('') + " " + df['Negative_Review'].fillna('')

# Fjern tomme anmeldelser
df = df[df['review_text'].str.strip() != '']

# Opprett sentiment-labels (1 = positiv, 0 = negativ)
df['sentiment'] = df.apply(lambda row: 1 if row['Review_Total_Positive_Word_Counts'] >= row['Review_Total_Negative_Word_Counts'] else 0, axis=1)

#Bruk kun et mindre datasett for raskere trening
df_sampled = df.sample(n=5000, random_state=42)

# Splitt i trenings- og testsett (80/20)
X_train, X_test, y_train, y_test = train_test_split(df_sampled['review_text'], df_sampled['sentiment'], test_size=0.2, random_state=42)

#Last inn DistilBERT-tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenizer-funksjon for å konvertere tekst til tokens
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=256, return_tensors="pt")

#Lag en PyTorch dataset-klasse
class HotelReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenize_function(texts.tolist())
        self.labels = torch.tensor(labels.values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# Konverter trenings- og testdata til PyTorch-datasett
train_dataset = HotelReviewDataset(X_train, y_train)
test_dataset = HotelReviewDataset(X_test, y_test)

#Last inn DistilBERT-modellen for klassifisering
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

#Definer treningsparametere (optimalisert for hastighet)
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    logging_dir="./logs",
    logging_steps=500
)


#Evalueringsfunksjon
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

#Tren modellen med Trainer-API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

#Evaluer modellen
results = trainer.evaluate()
print(results)
