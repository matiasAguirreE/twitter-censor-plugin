import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import os
import sys

# Add the project root to the Python path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local imports
from model.classifier import ToxicClassifier
from model.config_new import *
from model.device_utils import get_optimal_device, print_device_info, optimize_for_device, move_batch_to_device
from model.sentiment_analyzer import get_sentiment_analyzer, analyze_with_sentiment_correction

# --- Configuration & Setup ---
EVALUATION_DIR = os.path.join(ARTIFACTS_DIR, 'evaluation_results')
OUTPUT_FILE = os.path.join(EVALUATION_DIR, 'error_analysis.csv')
os.makedirs(EVALUATION_DIR, exist_ok=True)

device, device_name = print_device_info()

# --- Dataset Definition ---
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

# --- Evaluation Core Functions (from previous script) ---
def get_base_predictions(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    texts = []
    with torch.no_grad():
        for batch in data_loader:
            texts.extend(batch['text'])
            labels = batch['labels']
            batch = move_batch_to_device(batch, device)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            outputs = model(input_ids, attention_mask)
            preds = outputs.cpu().numpy()
            predictions.extend(preds)
            actual_labels.extend(labels.cpu().numpy())
    return texts, np.array(actual_labels), np.array(predictions)

def get_sa_corrected_predictions(texts, base_preds_probs):
    sa_corrected_preds = []
    for text, base_pred_prob in zip(texts, base_preds_probs):
        original_scores = {label: prob for label, prob in zip(LABELS, base_pred_prob)}
        enhanced_analysis = analyze_with_sentiment_correction(text, original_scores)
        corrected_scores = enhanced_analysis['corrected_toxicity']
        binary_preds = [1 if corrected_scores.get(label, 0) > 0.5 else 0 for label in LABELS]
        sa_corrected_preds.append(binary_preds)
    return np.array(sa_corrected_preds)

# --- Main Analysis Function ---
def analyze_and_compare():
    """Main function to run the error analysis."""
    print("🚀 Starting error analysis...")

    # 1. Load Model, Tokenizer, and SA
    print("📂 Loading model, tokenizer, and sentiment analyzer...")
    tokenizer = AutoTokenizer.from_pretrained(ARTIFACTS_DIR)
    model = ToxicClassifier(dropout=DROPOUT)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model = optimize_for_device(model, device)
    sa = get_sentiment_analyzer()
    print("✅ All components loaded.")

    # 2. Load and Prepare Data
    print("📂 Loading and splitting dataset...")
    df_test = pd.read_csv(DATA_TEST_PATH)
    X_val_test = df_test["Tweet"].tolist()
    y_val_test = df_test[LABELS].values.tolist()
    X_test, _, y_test, _ = train_test_split(X_val_test, y_val_test, test_size=0.4, random_state=42)
    test_dataset = TweetDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"🧪 Analyzing {len(X_test)} test samples.")

    # 3. Get Predictions
    print("🤖 Getting predictions from both models...")
    texts, y_true, y_pred_probs = get_base_predictions(model, test_loader, device)
    y_pred_base = (y_pred_probs > 0.5).astype(int)
    y_pred_sa = get_sa_corrected_predictions(texts, y_pred_probs)

    # 4. Compare and find disagreements
    print("🔍 Comparing predictions and identifying disagreements...")
    analysis_results = []
    for i in range(len(texts)):
        true_label = y_true[i]
        base_pred = y_pred_base[i]
        sa_pred = y_pred_sa[i]

        base_correct = np.array_equal(true_label, base_pred)
        sa_correct = np.array_equal(true_label, sa_pred)

        # Skip cases where both models agree (both right or both wrong)
        if base_correct == sa_correct:
            continue

        category = ""
        if not base_correct and sa_correct:
            category = "SA_Fixed_Error"
        elif base_correct and not sa_correct:
            category = "SA_Introduced_Error"
        
        analysis_results.append({
            "category": category,
            "text": texts[i],
            "true_labels": str(true_label.astype(int)),
            "base_prediction": str(base_pred),
            "sa_prediction": str(sa_pred)
        })

    # 5. Save results to CSV
    if not analysis_results:
        print("✅ No disagreements found between the models on this dataset.")
        return

    df_analysis = pd.DataFrame(analysis_results)
    df_analysis.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print("🎉 Error analysis complete!")
    print(f"📂 Report saved to: {OUTPUT_FILE}")
    print(f"   - {len(df_analysis[df_analysis['category'] == 'SA_Fixed_Error'])} cases where SA fixed an error.")
    print(f"   - {len(df_analysis[df_analysis['category'] == 'SA_Introduced_Error'])} cases where SA introduced an error.")
    print("="*50)

if __name__ == "__main__":
    analyze_and_compare()
