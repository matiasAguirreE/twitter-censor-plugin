import torch
import torch.nn as nn
from transformers import AutoModel
from config import MODEL_NAME, NUM_LABELS

# Define the ToxicClassifier model
class ToxicClassifier(nn.Module):
    def __init__(self):
        super(ToxicClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, NUM_LABELS)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.pooler_output
        dropped = self.dropout(pooled)
        return torch.sigmoid(self.classifier(dropped))
