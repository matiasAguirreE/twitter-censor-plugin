import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local imports
from model.classifier import ToxicClassifier
from model.config_new import *
from model.device_utils import get_optimal_device, print_device_info, optimize_for_device, move_batch_to_device
from model.sentiment_analyzer import get_sentiment_analyzer, analyze_with_sentiment_correction

# --- Configuration & Setup ---
EVALUATION_DIR = os.path.join(ARTIFACTS_DIR, 'evaluation_results')
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

# --- Evaluation Core Functions ---
def get_base_predictions(model, data_loader, device):
    """Gets raw predictions from the base model."""
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
            
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                preds = np.zeros(labels.shape)
            else:
                preds = (outputs.cpu() > CLASSIFICATION_THRESHOLD).numpy().astype(int)
            
            predictions.extend(preds)
            actual_labels.extend(labels.cpu().numpy())
            
    return texts, np.array(actual_labels), np.array(predictions)

def get_sa_corrected_predictions(texts, base_preds_probs):
    """Gets predictions corrected by the sentiment analyzer."""
    sa_corrected_preds = []
    for text, base_pred_prob in zip(texts, base_preds_probs):
        original_scores = {label: prob for label, prob in zip(LABELS, base_pred_prob)}
        enhanced_analysis = analyze_with_sentiment_correction(text, original_scores)
        corrected_scores = enhanced_analysis['corrected_toxicity']
        
        # Convert corrected scores to binary predictions
        binary_preds = [1 if corrected_scores.get(label, 0) > CLASSIFICATION_THRESHOLD else 0 for label in LABELS]
        sa_corrected_preds.append(binary_preds)
        
    return np.array(sa_corrected_preds)

def generate_evaluation_artifacts(y_true, y_pred, model_name):
    """Generates and saves classification report and confusion matrices."""
    print(f"--- Generating report for: {model_name} ---")
    
    # Create a dedicated directory for the model's results
    model_results_dir = os.path.join(EVALUATION_DIR, model_name)
    os.makedirs(model_results_dir, exist_ok=True)

    # 1. Classification Report
    report = classification_report(y_true, y_pred, target_names=LABELS, zero_division=0)
    report_path = os.path.join(model_results_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"📊 Classification report saved to: {report_path}")
    print(report)

    # 2. Confusion Matrices per Class
    for i, label in enumerate(LABELS):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted Negative', 'Predicted Positive'], 
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title(f'Confusion Matrix for {label} ({model_name})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_path = os.path.join(model_results_dir, f'cm_{label}.png')
        plt.savefig(cm_path)
        plt.close() # Close the plot to avoid displaying it
        print(f"🖼️ Confusion matrix for {label} saved to: {cm_path}")

    # 3. Overall "Censurable" vs "Incensurable" Matrix
    y_true_censurable = y_true.any(axis=1)
    y_pred_censurable = y_pred.any(axis=1)
    
    cm_overall = confusion_matrix(y_true_censurable, y_pred_censurable)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_overall, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Predicted Incensurable', 'Predicted Censurable'],
                yticklabels=['Actual Incensurable', 'Actual Censurable'])
    plt.title(f'Overall Censorship Matrix ({model_name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_overall_path = os.path.join(model_results_dir, 'cm_overall_censorship.png')
    plt.savefig(cm_overall_path)
    plt.close()
    print(f"🖼️ Overall censorship matrix saved to: {cm_overall_path}")


def main():
    """Main function to run the evaluation."""
    print("🚀 Starting model evaluation...")

    # 1. Load Model and Tokenizer
    print(f"📂 Loading model from: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(ARTIFACTS_DIR)
        model = ToxicClassifier(dropout=DROPOUT)
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model = optimize_for_device(model, device)
        model.eval()
        print("✅ Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Load Sentiment Analyzer
    print("🧠 Initializing Sentiment Analyzer...")
    sa = get_sentiment_analyzer()
    if not sa.is_available():
        print("❌ Sentiment analyzer not available. Cannot perform SA-corrected evaluation.")
        return
    print("✅ Sentiment analyzer initialized.")

    # 3. Load and Prepare Data
    print("📂 Loading and splitting dataset...")
    df_test = pd.read_csv(DATA_TEST_PATH)
    X_val_test = df_test["Tweet"].tolist()
    y_val_test = df_test[LABELS].values.tolist()

    # Replicate the exact split from training
    X_test, _, y_test, _ = train_test_split(X_val_test, y_val_test, test_size=0.4, random_state=42)
    
    print(f"🧪 Using {len(X_test)} samples for testing.")
    
    test_dataset = TweetDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Get Predictions
    print("🤖 Getting predictions from base model...")
    texts, y_true, y_pred_probs = get_base_predictions(model, test_loader, device)
    y_pred_base = (y_pred_probs > 0.5).astype(int)

    print("🧠 Getting SA-corrected predictions...")
    y_pred_sa_corrected = get_sa_corrected_predictions(texts, y_pred_probs)

    # 5. Generate all artifacts
    print("\n" + "="*50)
    generate_evaluation_artifacts(y_true, y_pred_base, "model_base")
    print("\n" + "="*50)
    generate_evaluation_artifacts(y_true, y_pred_sa_corrected, "model_with_sa")
    print("\n" + "="*50)
    
    print("🎉 Evaluation complete!")
    print(f"📂 All results saved in: {EVALUATION_DIR}")

if __name__ == "__main__":
    main()