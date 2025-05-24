import torch
from .classifier import ToxicClassifier
from .config import *
from transformers import AutoTokenizer

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to predict the toxicity of a given text
def predict(text, model, device, tokenizer):
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

if __name__ == "__main__":
    #cargar el tokenizador
    tokenizer = AutoTokenizer.from_pretrained(ARTIFACTS_DIR)

    # Carga el modelo UNA sola vez aquí
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ToxicClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.to(device)
    model.eval()
    
    text = input("Enter text to predict: ")
    pred = predict(text, model, device, tokenizer)
    print(pred)
