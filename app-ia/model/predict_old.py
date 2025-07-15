import torch
from .classifier import ToxicClassifier
from .config_old import *
from .device_utils import get_optimal_device, optimize_for_device
from transformers import AutoTokenizer

# Get optimal device with MPS support
device, device_name = get_optimal_device()

# Function to predict the toxicity of a given text using OLD MODEL
def predict_old(text, model, device, tokenizer):
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

def load_old_model():
    """Load OLD model with MPS optimization"""
    try:
        print(f"🤖 Loading OLD model from: {MODEL_PATH}")
        print(f"🎯 Target device: {device_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(ARTIFACTS_DIR)
        
        # Load model
        model = ToxicClassifier(dropout=DROPOUT)
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        
        # Apply device optimizations
        model = optimize_for_device(model, device)
        model.eval()
        
        print(f"✅ OLD model loaded successfully on {device}")
        return model, tokenizer, device
        
    except FileNotFoundError:
        print(f"❌ OLD model not found at {MODEL_PATH}")
        print("💡 The old model should be available. Check the model path.")
        return None, None, None
    except Exception as e:
        print(f"❌ Error loading OLD model: {e}")
        return None, None, None

if __name__ == "__main__":
    model, tokenizer, device = load_old_model()
    if model is not None:
        text = input("Enter text to predict with OLD model: ")
        pred = predict_old(text, model, device, tokenizer)
        print(pred)
    else:
        print("Failed to load old model") 