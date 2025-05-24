import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from model import ToxicClassifier
from config import *
from sklearn.model_selection import train_test_split
import os

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prtin para ver que version de CUDA se usa
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (compiled):", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# Define the TweetDataset class for loading and processing the dataset
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }

def train():
    # Load the dataset with tweets and labels
    df = pd.read_csv(DATA_PATH)
    X = df["Tweet"].tolist()
    y = df[LABELS].values.tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    train_ds = TweetDataset(X_train, y_train, tokenizer)
    val_ds = TweetDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = ToxicClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    tokenizer.save_pretrained(ARTIFACTS_DIR)

if __name__ == "__main__":
    train()
