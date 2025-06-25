# -*- coding: utf-8 -*- 
from app import app
import torch
import json
from functools import wraps
from collections import OrderedDict
from flask import Flask, jsonify, request
from transformers import AutoTokenizer

# Local imports
from model import predict_old, predict_new, sentiment_analyzer
from utils import compare_predictions

# --- Model and Analyzer Initialization ---

print("Loading OLD model...")
old_model, old_tokenizer, old_device = predict_old.load_old_model()
old_model_available = old_model is not None
print(f"✓ OLD model loaded: {old_model_available}")

print("Loading NEW model...")
new_model, new_tokenizer, new_device = predict_new.load_new_model()
new_model_available = new_model is not None
print(f"✓ NEW model loaded: {new_model_available}")

print("Initializing Sentiment Analyzer...")
sa = sentiment_analyzer.get_sentiment_analyzer()
sentiment_analyzer_available = sa.is_available()
print(f"✓ Sentiment analyzer initialized: {sentiment_analyzer_available}")

# --- Decorators for Security ---

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-Api-Key')
        if not api_key or api_key != 'e55d7f49-a705-4895-bf5f-d63aa1f46e11':
            return jsonify({"error": "Unauthorized: Invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated_function

# --- API Endpoints ---

@app.route('/status/', methods=['GET'])
def status():
    """Endpoint to check which models and analyzers are available."""
    return jsonify({
        'old_model_available': old_model_available,
        'new_model_available': new_model_available,
        'sentiment_analyzer_available': sentiment_analyzer_available,
        'total_models_loaded': sum([old_model_available, new_model_available])
    })

# --- Legacy Endpoint for Backward Compatibility ---

@app.route('/verificarCensura/', methods=['POST'])
def verificarCensura():
    """Legacy endpoint now using NEW model with optional sentiment analysis"""
    api_key = request.headers.get('X-Api-Key')
    if api_key != 'e55d7f49-a705-4895-bf5f-d63aa1f46e11':
        return jsonify({'error': 'No autorizado'}), 401

    tweet = request.json
    print("se recibe: {}".format(tweet))
    
    texto = tweet['texto']
    
    # Check for optional sentiment analysis parameter (default: False)
    apply_sa = tweet.get('sa', False)
    
    if apply_sa:
        # Use NEW model with sentiment analysis correction
        if not new_model_available:
            return jsonify({'error': 'Modelo nuevo no disponible'}), 500
        
        if not sentiment_analyzer_available:
            return jsonify({'error': 'Analizador de sentimiento no disponible'}), 500
        
        # Get original scores from NEW model
        original_scores = predict_new.predict_new(texto, new_model, new_device, new_tokenizer)
        
        # Apply sentiment analysis and correction
        enhanced_analysis = sentiment_analyzer.analyze_with_sentiment_correction(texto, original_scores)
        
        # Get corrected toxicity scores
        corrected_scores = enhanced_analysis['corrected_toxicity']
        
        # Get sentiment information
        sentiment_info = enhanced_analysis.get('sentiment_analysis', {})
        
        # Build response in the requested format with specific order
        response = OrderedDict([
            ('Homofobia', corrected_scores.get('Homofobia', 0)),
            ('Violencia', corrected_scores.get('Violencia', 0)),
            ('Xenofobia', corrected_scores.get('Xenofobia', 0)),
            ('Sentiment', sentiment_info.get('label', 'Unknown')),
            ('Sentiment_prob', sentiment_info.get('confidence', 0))
        ])
        
        print("se envia (con SA): {}".format(response))
        
        # Use custom response to preserve field order
        response_json = json.dumps(response, ensure_ascii=False)
        flask_response = app.response_class(
            response=response_json,
            status=200,
            mimetype='application/json'
        )
        return flask_response
    
    else:
        # Standard behavior: use NEW model without sentiment analysis
        if not new_model_available:
            return jsonify({'error': 'Modelo nuevo no disponible'}), 500
        
        censura = predict_new.predict_new(texto, new_model, new_device, new_tokenizer)
        print("se envia: {}".format(censura))
        return jsonify(censura)

# --- Basic Toxicity Endpoints ---

@app.route('/verificarCensura/old/', methods=['POST'])
@require_api_key
def verify_censorship_old():
    """Endpoint for OLD model predictions."""
    data = request.get_json()
    text = data.get('text', '') or data.get('texto', '')  # Support both formats
    if not text:
        return jsonify({"error": "Text is required"}), 400
    
    if not old_model_available:
        return jsonify({"error": "Old model is not available"}), 503
    
    prediction = predict_old.predict_old(text, old_model, old_device, old_tokenizer)
    return jsonify(prediction)

@app.route('/verificarCensura/new/', methods=['POST'])
@require_api_key
def verify_censorship_new():
    """Endpoint for NEW model predictions."""
    data = request.get_json()
    text = data.get('text', '') or data.get('texto', '')  # Support both formats
    if not text:
        return jsonify({"error": "Text is required"}), 400
    
    if not new_model_available:
        return jsonify({"error": "New model is not available"}), 503
    
    prediction = predict_new.predict_new(text, new_model, new_device, new_tokenizer)
    return jsonify(prediction)

@app.route('/verificarCensura/compare/', methods=['POST'])
@require_api_key
def compare_censorship():
    """Endpoint to compare raw predictions from both models."""
    data = request.get_json()
    text = data.get('text', '') or data.get('texto', '')  # Support both formats
    if not text:
        return jsonify({"error": "Text is required"}), 400

    # Old model
    old_model_response = {"available": False, "prediction": {}}
    if old_model_available:
        old_prediction = predict_old.predict_old(text, old_model, old_device, old_tokenizer)
        old_model_response = {"available": True, "prediction": old_prediction}

    # New model
    new_model_response = {"available": False, "prediction": {}}
    if new_model_available:
        new_prediction = predict_new.predict_new(text, new_model, new_device, new_tokenizer)
        new_model_response = {"available": True, "prediction": new_prediction}
        
    comparison = compare_predictions(
        old_model_response.get('prediction', {}), 
        new_model_response.get('prediction', {})
    )
    
    return jsonify({
        "text": text,
        "old_model_raw": old_model_response,
        "new_model_raw": new_model_response,
        "comparison": comparison
    })

# --- Endpoints with Sentiment Analysis (SA) Correction ---

@app.route('/verificarCensura/old/sa/', methods=['POST'])
@require_api_key
def verify_censorship_old_sa():
    """Endpoint for old model with sentiment analysis correction."""
    data = request.get_json()
    text = data.get('text', '') or data.get('texto', '')  # Support both formats
    if not text:
        return jsonify({"error": "Text is required"}), 400
    
    if not old_model_available or not sentiment_analyzer_available:
        return jsonify({"error": "Required model or analyzer is not available"}), 503

    original_scores = predict_old.predict_old(text, old_model, old_device, old_tokenizer)
    enhanced_analysis = sentiment_analyzer.analyze_with_sentiment_correction(text, original_scores)
    
    return jsonify(enhanced_analysis['corrected_toxicity'])

@app.route('/verificarCensura/new/sa/', methods=['POST'])
@require_api_key
def verify_censorship_new_sa():
    """Endpoint for new model with sentiment analysis correction."""
    data = request.get_json()
    text = data.get('text', '') or data.get('texto', '')  # Support both formats
    if not text:
        return jsonify({"error": "Text is required"}), 400
    
    if not new_model_available or not sentiment_analyzer_available:
        return jsonify({"error": "Required model or analyzer is not available"}), 503
        
    original_scores = predict_new.predict_new(text, new_model, new_device, new_tokenizer)
    enhanced_analysis = sentiment_analyzer.analyze_with_sentiment_correction(text, original_scores)
    
    return jsonify(enhanced_analysis['corrected_toxicity'])

@app.route('/verificarCensura/compare/sa/', methods=['POST'])
@require_api_key
def compare_censorship_sa():
    """Endpoint to compare NEW model with and without sentiment analysis correction."""
    data = request.get_json()
    text = data.get('text', '') or data.get('texto', '')  # Support both formats
    if not text:
        return jsonify({"error": "Text is required"}), 400

    if not new_model_available:
        return jsonify({"error": "New model is not available"}), 503

    if not sentiment_analyzer_available:
        return jsonify({"error": "Sentiment analyzer is not available"}), 503

    # NEW model WITHOUT SA (original scores)
    new_original = predict_new.predict_new(text, new_model, new_device, new_tokenizer)
    new_without_sa_response = {"available": True, "prediction": new_original}

    # NEW model WITH SA (corrected scores)
    new_corrected = sentiment_analyzer.analyze_with_sentiment_correction(text, new_original)['corrected_toxicity']
    new_with_sa_response = {"available": True, "prediction": new_corrected}
        
    comparison = compare_predictions(
        new_without_sa_response.get('prediction', {}), 
        new_with_sa_response.get('prediction', {})
    )
    
    # Use OrderedDict to ensure consistent field ordering
    response_data = OrderedDict([
        ("text", text),
        ("new_model_without_sa", new_without_sa_response),
        ("new_model_with_sa", new_with_sa_response),
        ("comparison_sa_impact", comparison)
    ])
    
    return jsonify(response_data)

# --- Standalone Sentiment Analysis Endpoint ---

@app.route('/sentiment/', methods=['POST'])
@require_api_key
def sentiment_analysis_endpoint():
    """Endpoint for sentiment analysis only."""
    data = request.get_json()
    text = data.get('text', '') or data.get('texto', '')  # Support both formats
    if not text:
        return jsonify({"error": "Text is required"}), 400
    
    if not sentiment_analyzer_available:
        return jsonify({"error": "Sentiment analyzer is not available"}), 503
        
    sentiment_result = sa.predict_sentiment(text)
    return jsonify({"text": text, "sentiment": sentiment_result})

# --- Legacy Enhanced Endpoints (kept for compatibility) ---

@app.route('/verificarCensura/enhanced/', methods=['POST'])
@require_api_key
def verify_censorship_enhanced_auto():
    """
    Enhanced endpoint with sentiment analysis and bias correction.
    Uses NEW model by default, falls back to OLD model if available.
    Returns detailed analysis object.
    """
    data = request.get_json()
    text = data.get('text', '') or data.get('texto', '')  # Support both formats
    if not text:
        return jsonify({"error": "Text is required"}), 400
        
    if not sentiment_analyzer_available:
        return jsonify({"error": "Sentiment analyzer is not available"}), 503
    
    analysis = None
    model_used = None
    if new_model_available:
        original_scores = predict_new.predict_new(text, new_model, new_device, new_tokenizer)
        analysis = sentiment_analyzer.analyze_with_sentiment_correction(text, original_scores)
        model_used = "new"
    elif old_model_available:
        original_scores = predict_old.predict_old(text, old_model, old_device, old_tokenizer)
        analysis = sentiment_analyzer.analyze_with_sentiment_correction(text, original_scores)
        model_used = "old"
    else:
        return jsonify({'error': 'No models available'}), 503
    
    analysis['model_used'] = model_used
    analysis['text'] = text
    
    return jsonify(analysis)


if __name__ == "__main__":
    print("\n🚀 Server starting...")
    print(f"   - OLD model: {'✓ Available' if old_model_available else '✗ Not available'}")
    print(f"   - NEW model: {'✓ Available' if new_model_available else '✗ Not available'}")
    print(f"   - Sentiment analyzer: {'✓ Available' if sentiment_analyzer_available else '✗ Not available'}")
    
    print("\n🌐 API Endpoints Ready")
    app.run(host='0.0.0.0', port=7021)
