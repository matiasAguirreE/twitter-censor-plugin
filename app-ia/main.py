# -*- coding: utf-8 -*- 
from app import app
from utils import verificarCensura_
from flask import Flask, jsonify, request

@app.route('/verificarCensura/', methods=['POST'])
def verificarCensura():
    api_key = request.headers.get('X-Api-Key')
    if api_key != 'e55d7f49-a705-4895-bf5f-d63aa1f46e11':
        return jsonify({'error': 'No autorizado'}), 401

    tweet = request.json
    print("se recibe: {}".format(tweet))
    texto = tweet['texto']
    censura = verificarCensura_(texto)
    print("se envia: {}".format(censura))
    return censura

if __name__ == "__main__":
    app.run(port=7021)
