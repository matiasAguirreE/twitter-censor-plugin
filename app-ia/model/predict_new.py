import torch
from .classifier import ToxicClassifier
from .config_new import *
from .device_utils import get_optimal_device, optimize_for_device
from transformers import AutoTokenizer

# Get optimal device with MPS support
device, device_name = get_optimal_device()

# Function to predict the toxicity of a given text using NEW MODEL
def predict_new(text, model, device, tokenizer):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LEN)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = outputs.cpu().squeeze().numpy()
        result = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
        return result

def load_new_model():
    """Load NEW model with MPS optimization"""
    try:
        print(f"🤖 Loading NEW model from: {MODEL_PATH}")
        print(f"🎯 Target device: {device_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(ARTIFACTS_DIR)
        
        # Load model
        model = ToxicClassifier(dropout=DROPOUT)
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        
        # Apply device optimizations
        model = optimize_for_device(model, device)
        model.eval()
        
        print(f"✅ NEW model loaded successfully on {device}")
        return model, tokenizer, device
        
    except FileNotFoundError:
        print(f"❌ NEW model not found at {MODEL_PATH}")
        print("💡 Train the new model first using: python train_new_model.py")
        return None, None, None
    except Exception as e:
        print(f"❌ Error loading NEW model: {e}")
        return None, None, None

if __name__ == "__main__":
    model, tokenizer, device = load_new_model()
    if model is not None:
        text = input("Enter text to predict with NEW model: ")
        pred = predict_new(text, model, device, tokenizer)
        print(pred)
    else:
        print("Failed to load new model") 