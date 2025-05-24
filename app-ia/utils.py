# -*- coding: utf-8 -*- 
from model.predict import predict

def verificarCensura_(texto, model, device, tokenizer):
    return predict(texto, model, device, tokenizer)
