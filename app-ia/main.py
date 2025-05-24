# -*- coding: utf-8 -*- 
from app import app
import torch
from utils import verificarCensura_
from flask import Flask, jsonify, request
from transformers import AutoTokenizer
from model.config import MODEL_PATH, ARTIFACTS_DIR
from model.classifier import ToxicClassifier

#cargar el tokenizador
tokenizer = AutoTokenizer.from_pretrained(ARTIFACTS_DIR)

# Carga el modelo UNA sola vez aquí
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ToxicClassifier()
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.to(device)
model.eval()


@app.route('/verificarCensura/', methods=['POST'])
def verificarCensura():
    api_key = request.headers.get('X-Api-Key')
    if api_key != 'e55d7f49-a705-4895-bf5f-d63aa1f46e11':
        return jsonify({'error': 'No autorizado'}), 401

    tweet = request.json
    print("se recibe: {}".format(tweet))
    texto = tweet['texto']
    censura = verificarCensura_(texto, model, device, tokenizer)
    print("se envia: {}".format(censura))
    return censura

if __name__ == "__main__":
    app.run(port=7021)
