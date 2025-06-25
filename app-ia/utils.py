# -*- coding: utf-8 -*- 
from model.predict import predict
from model.predict_old import predict_old
from model.predict_new import predict_new

def verificarCensura_(texto, model, device, tokenizer):
    """Original function for backward compatibility"""
    return predict(texto, model, device, tokenizer)

def verificarCensura_old(texto, model, device, tokenizer):
    """Function to verify censorship using OLD model"""
    return predict_old(texto, model, device, tokenizer)

def verificarCensura_new(texto, model, device, tokenizer):
    """Function to verify censorship using NEW model"""
    return predict_new(texto, model, device, tokenizer)
