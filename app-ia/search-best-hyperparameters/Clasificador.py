import torch.nn as nn
from transformers import AutoModel

  

class TweetClasificador(nn.Module):
    def __init__(self,dropout:float):
        super(TweetClasificador,self).__init__()

        self.bert=AutoModel.from_pretrained("dccuchile/tulio-chilean-spanish-bert", from_tf=False)
        self.dropout=nn.Dropout(dropout)
        self.classifier=nn.Linear(self.bert.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        salida=self.bert(input_ids=input_ids,attention_mask=attention_mask)
        pooled=salida.pooler_output
        dropped=self.dropout(pooled)
        logits=self.classifier(dropped)
        return logits