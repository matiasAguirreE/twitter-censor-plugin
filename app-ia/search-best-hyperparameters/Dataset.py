from torch.utils.data import Dataset
import torch

class TweetDataset(Dataset):
    def __init__(self,tweets,etiquetas,tokenizador):
        self.tweets=tweets
        self.etiquetas=etiquetas
        self.tokenizador=tokenizador

    def __len__(self): return len(self.tweets)

    def __getitem__(self,idx):
        enc=self.tokenizador(
            self.tweets.iloc[idx],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        label_row=self.etiquetas.iloc[idx].values
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_row,dtype=torch.float)
        }