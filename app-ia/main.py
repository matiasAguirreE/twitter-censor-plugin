# -*- coding: utf-8 -*- 
from app import app
from utils import verificarCensura_
from flask import Flask, jsonify, request

@app.route('/verificarCensura', methods=['POST'])
def verificarCensura():
    tweet = request.json
    print("se recibe: {}".format(tweet))
    texto = tweet['texto']
    censura = verificarCensura_(texto)
    print("se envia: {}".format(censura))
    return {'censura': censura}

if __name__ == "__main__":
    app.run(port=7021)
