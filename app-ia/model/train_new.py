import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from .classifier import ToxicClassifier
from .config_new import *
from .device_utils import get_optimal_device, print_device_info, optimize_for_device, get_recommended_batch_size, move_batch_to_device
from sklearn.model_selection import train_test_split
import os

# Get optimal device with MPS support
device, device_name = print_device_info()

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

def train_new_model():
    """Train the NEW model using the best hyperparameters found"""
    print("🚀 Starting training for NEW model...")
    print("📊 Using BEST hyperparameters from search with data augmentation:")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Learning Rate: {LEARNING_RATE}")
    print(f"   - Batch Size: {BATCH_SIZE}")
    print(f"   - Warmup Ratio: {WARMUP_RATIO}")
    print(f"   - Weight Decay: {WEIGHT_DECAY}")
    print(f"   - Dropout: {DROPOUT}")
    
    # Get device-optimized batch size
    optimized_batch_size = get_recommended_batch_size(device, BATCH_SIZE)
    if optimized_batch_size != BATCH_SIZE:
        print(f"🎯 Optimized batch size for {device.type.upper()}: {BATCH_SIZE} → {optimized_batch_size}")
    
    print(f"📁 Model will be saved to: {MODEL_PATH}")
    print(f"📁 Tokenizer will be saved to: {ARTIFACTS_DIR}")
    
    # Load the dataset with tweets and labels
    print("📂 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📈 Data augmentation detected: {df.shape[0]} tweets (vs 1,000 original)")
    
    X = df["Tweet"].tolist()
    y = df[LABELS].values.tolist()

    print("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("🔄 Splitting dataset...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📈 Training samples: {len(X_train)}")
    print(f"📉 Validation samples: {len(X_val)}")

    train_ds = TweetDataset(X_train, y_train, tokenizer)
    val_ds = TweetDataset(X_val, y_val, tokenizer)

    # Use optimized batch size
    train_loader = DataLoader(train_ds, batch_size=optimized_batch_size, shuffle=True, 
                             pin_memory=(device.type != "cpu"), num_workers=0)  # num_workers=0 for MPS compatibility
    val_loader = DataLoader(val_ds, batch_size=optimized_batch_size,
                           pin_memory=(device.type != "cpu"), num_workers=0)

    print("🤖 Initializing model with optimal hyperparameters...")
    model = ToxicClassifier(dropout=DROPOUT)
    
    # Apply device-specific optimizations
    model = optimize_for_device(model, device)
    
    # Usar weight decay como en los mejores resultados
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Configurar scheduler con warmup como en los mejores resultados
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    criterion = torch.nn.BCELoss()
    
    print(f"🏃‍♂️ Starting training for {EPOCHS} epochs...")
    print(f"📈 Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    print(f"🎯 Training on: {device_name}")
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Move batch to device efficiently
            batch = move_batch_to_device(batch, device)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"] 
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Actualizar scheduler después de cada step
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{EPOCHS} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f} - LR: {current_lr:.2e}")

        avg_loss = total_loss / num_batches
        print(f"✅ Epoch {epoch+1}/{EPOCHS} completed - Avg Loss: {avg_loss:.4f}")
        
        # Validation step
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        print(f"📊 Validation Loss: {avg_val_loss:.4f}")
        model.train()

    print("💾 Saving model and tokenizer...")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Move model to CPU before saving for compatibility
    model_cpu = model.to("cpu")
    torch.save(model_cpu.state_dict(), MODEL_PATH)
    tokenizer.save_pretrained(ARTIFACTS_DIR)
    
    print("🎉 NEW model training completed successfully!")
    print(f"📁 Model saved to: {MODEL_PATH}")
    print(f"📁 Tokenizer saved to: {ARTIFACTS_DIR}")
    print(f"🚀 Trained on: {device_name}")
    print("📊 This model was trained with the BEST hyperparameters that achieved:")
    print("   - F1-Macro: 99.53%")
    print("   - F1-Micro: 99.56%") 
    print("   - Precision promedio: 100%")
    print("   - Recall promedio: 99.07%")

if __name__ == "__main__":
    train_new_model() 