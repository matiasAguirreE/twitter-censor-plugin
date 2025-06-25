# -*- coding: utf-8 -*- 
from app import app
import torch
import json
from utils import verificarCensura_, verificarCensura_old, verificarCensura_new
from flask import Flask, jsonify, request
from transformers import AutoTokenizer
from model.predict_old import load_old_model
from model.predict_new import load_new_model

# Load OLD model
print("Loading OLD model...")
old_model, old_tokenizer, old_device = load_old_model()
if old_model is not None:
    print("✓ OLD model loaded successfully")
else:
    print("✗ Failed to load OLD model")

# Load NEW model
print("Loading NEW model...")
new_model, new_tokenizer, new_device = load_new_model()
if new_model is not None:
    print("✓ NEW model loaded successfully")
else:
    print("✗ Failed to load NEW model (this is expected if not trained yet)")

@app.route('/verificarCensura/', methods=['POST'])
def verificarCensura():
    """Original endpoint using OLD model for backward compatibility"""
    api_key = request.headers.get('X-Api-Key')
    if api_key != 'e55d7f49-a705-4895-bf5f-d63aa1f46e11':
        return jsonify({'error': 'No autorizado'}), 401

    if old_model is None:
        return jsonify({'error': 'Modelo viejo no disponible'}), 500

    tweet = request.json
    print("se recibe: {}".format(tweet))
    texto = tweet['texto']
    censura = verificarCensura_old(texto, old_model, old_device, old_tokenizer)
    print("se envia: {}".format(censura))
    return censura

@app.route('/verificarCensura/old/', methods=['POST'])
def verificarCensura_old_endpoint():
    """Endpoint for OLD model predictions - maintains same format as original"""
    api_key = request.headers.get('X-Api-Key')
    if api_key != 'e55d7f49-a705-4895-bf5f-d63aa1f46e11':
        return jsonify({'error': 'No autorizado'}), 401

    if old_model is None:
        return jsonify({'error': 'Modelo viejo no disponible'}), 500

    tweet = request.json
    print("OLD MODEL - se recibe: {}".format(tweet))
    texto = tweet['texto']
    censura = verificarCensura_old(texto, old_model, old_device, old_tokenizer)
    print("OLD MODEL - se envia: {}".format(censura))
    return censura

@app.route('/verificarCensura/new/', methods=['POST'])
def verificarCensura_new_endpoint():
    """Endpoint for NEW model predictions - maintains same format as original"""
    api_key = request.headers.get('X-Api-Key')
    if api_key != 'e55d7f49-a705-4895-bf5f-d63aa1f46e11':
        return jsonify({'error': 'No autorizado'}), 401

    if new_model is None:
        return jsonify({'error': 'Modelo nuevo no disponible'}), 500

    tweet = request.json
    print("NEW MODEL - se recibe: {}".format(tweet))
    texto = tweet['texto']
    censura = verificarCensura_new(texto, new_model, new_device, new_tokenizer)
    print("NEW MODEL - se envia: {}".format(censura))
    return censura

@app.route('/verificarCensura/compare/', methods=['POST'])
def verificarCensura_compare():
    """Endpoint to compare predictions from both models"""
    api_key = request.headers.get('X-Api-Key')
    if api_key != 'e55d7f49-a705-4895-bf5f-d63aa1f46e11':
        return jsonify({'error': 'No autorizado'}), 401

    tweet = request.json
    print("COMPARE - se recibe: {}".format(tweet))
    texto = tweet['texto']
    
    result = {
        'text': texto,
        'old_model': None,
        'new_model': None,
        'comparison': None
    }

    # Get prediction from OLD model
    if old_model is not None:
        try:
            old_prediction = verificarCensura_old(texto, old_model, old_device, old_tokenizer)
            result['old_model'] = {
                'available': True,
                'prediction': old_prediction
            }
        except Exception as e:
            result['old_model'] = {
                'available': False,
                'error': str(e)
            }
    else:
        result['old_model'] = {
            'available': False,
            'error': 'Modelo no cargado'
        }

    # Get prediction from NEW model
    if new_model is not None:
        try:
            new_prediction = verificarCensura_new(texto, new_model, new_device, new_tokenizer)
            result['new_model'] = {
                'available': True,
                'prediction': new_prediction
            }
        except Exception as e:
            result['new_model'] = {
                'available': False,
                'error': str(e)
            }
    else:
        result['new_model'] = {
            'available': False,
            'error': 'Modelo no cargado'
        }

    # Add comparison if both models are available
    if result['old_model']['available'] and result['new_model']['available']:
        old_pred = result['old_model']['prediction']
        new_pred = result['new_model']['prediction']
        
        comparison = {}
        for label in old_pred.keys():
            old_val = old_pred[label]
            new_val = new_pred[label]
            comparison[label] = {
                'old': old_val,
                'new': new_val,
                'difference': new_val - old_val,
                'percent_change': ((new_val - old_val) / max(old_val, 0.001)) * 100
            }
        
        result['comparison'] = comparison

    print("COMPARE - se envia: {}".format(result))
    # Return with better formatting for readability
    response = app.response_class(
        response=json.dumps(result, indent=2, ensure_ascii=False),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/status/', methods=['GET'])
def status():
    """Endpoint to check which models are available"""
    return jsonify({
        'old_model_available': old_model is not None,
        'new_model_available': new_model is not None,
        'total_models_loaded': sum([old_model is not None, new_model is not None])
    })

if __name__ == "__main__":
    print(f"\n🚀 Server starting...")
    print(f"📊 Models status:")
    print(f"   - OLD model: {'✓ Available' if old_model is not None else '✗ Not available'}")
    print(f"   - NEW model: {'✓ Available' if new_model is not None else '✗ Not available'}")
    print(f"\n🌐 Available endpoints:")
    print(f"   - POST /verificarCensura/ (backward compatibility - uses old model)")
    print(f"   - POST /verificarCensura/old/ (old model)")
    print(f"   - POST /verificarCensura/new/ (new model)")
    print(f"   - POST /verificarCensura/compare/ (compare both models)")
    print(f"   - GET  /status/ (check models availability)")
    print(f"\n🔑 All endpoints require X-Api-Key header")
    print(f"📡 Server running on port 7021")
    app.run(port=7021)
