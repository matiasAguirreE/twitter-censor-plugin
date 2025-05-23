import torch
import pandas as pd
from model import ToxicClassifier
from config import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from train import TweetDataset
from sklearn.metrics import f1_score, precision_score, recall_score

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to evaluate the model
def evaluate():
    # Load the dataset with tweets and labels
    df = pd.read_csv(DATA_PATH)
    texts = df["Tweet"].tolist()
    labels = df[LABELS].values.tolist()

    # Load the tokenizer and create a DataLoader for the dataset
    tokenizer = AutoTokenizer.from_pretrained(ARTIFACTS_DIR)
    dataset = TweetDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # Load the pre-trained model
    model = ToxicClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Evaluate the model on the dataset
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].cpu().numpy()
            outputs = model(input_ids, attention_mask).cpu().numpy()
            preds.extend(outputs)
            trues.extend(labels_batch)

    # Convert predictions and true labels to tensors
    preds_bin = (torch.tensor(preds) > 0.5).int()
    trues = torch.tensor(trues).int()

    # Calculate and print F1, Precision, and Recall scores for each label
    for i, label in enumerate(LABELS):
        f1 = f1_score(trues[:, i], preds_bin[:, i])
        precision = precision_score(trues[:, i], preds_bin[:, i])
        recall = recall_score(trues[:, i], preds_bin[:, i])
        print(f"{label}: F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

if __name__ == "__main__":
    evaluate()
