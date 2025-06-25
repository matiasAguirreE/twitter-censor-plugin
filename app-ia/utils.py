# -*- coding: utf-8 -*- 
from model.predict import predict
from model.predict_old import predict_old as predict_old_func
from model.predict_new import predict_new as predict_new_func
from model.sentiment_analyzer import analyze_with_sentiment_correction, get_sentiment_analyzer

def verificarCensura_(texto, model, device, tokenizer):
    """Original function for backward compatibility"""
    return predict(texto, model, device, tokenizer)

def verificarCensura_old(texto, model, device, tokenizer):
    """Function to verify censorship using OLD model"""
    return predict_old_func(texto, model, device, tokenizer)

def verificarCensura_new(texto, model, device, tokenizer):
    """Function to verify censorship using NEW model"""
    return predict_new_func(texto, model, device, tokenizer)

def verificarCensura_with_sentiment(texto, model, device, tokenizer, model_type="new"):
    """
    Verify censorship with sentiment analysis and bias correction
    
    Args:
        texto: Text to analyze
        model: Toxicity detection model
        device: Device for inference
        tokenizer: Model tokenizer
        model_type: "old" or "new" to specify which prediction function to use
        
    Returns:
        Dict with complete analysis including sentiment and bias correction
    """
    # Get toxicity prediction based on model type
    if model_type == "old":
        toxicity_scores = predict_old_func(texto, model, device, tokenizer)
    else:
        toxicity_scores = predict_new_func(texto, model, device, tokenizer)
    
    # Perform sentiment analysis and bias correction
    complete_analysis = analyze_with_sentiment_correction(texto, toxicity_scores)
    
    return complete_analysis

def verificarCensura_new_enhanced(texto, model, device, tokenizer):
    """
    Enhanced NEW model prediction with sentiment analysis and bias correction
    
    Args:
        texto: Text to analyze
        model: NEW toxicity detection model
        device: Device for inference
        tokenizer: Model tokenizer
        
    Returns:
        Dict with enhanced analysis including sentiment and bias correction
    """
    return verificarCensura_with_sentiment(texto, model, device, tokenizer, model_type="new")

def verificarCensura_old_enhanced(texto, model, device, tokenizer):
    """
    Enhanced OLD model prediction with sentiment analysis and bias correction
    
    Args:
        texto: Text to analyze
        model: OLD toxicity detection model
        device: Device for inference
        tokenizer: Model tokenizer
        
    Returns:
        Dict with enhanced analysis including sentiment and bias correction
    """
    return verificarCensura_with_sentiment(texto, model, device, tokenizer, model_type="old")

def get_sentiment_only(texto):
    """
    Get only sentiment analysis for a given text
    
    Args:
        texto: Text to analyze
        
    Returns:
        Dict with sentiment analysis or None if not available
    """
    analyzer = get_sentiment_analyzer()
    return analyzer.predict_sentiment(texto)

def compare_predictions(prediction1, prediction2):
    """
    Compare predictions from two models and calculate differences and percentage changes
    
    Args:
        prediction1: First prediction dictionary (e.g., from old model)
        prediction2: Second prediction dictionary (e.g., from new model)
        
    Returns:
        Dict with comparison data for each toxicity category
    """
    if not prediction1 or not prediction2:
        return {}
    
    comparison = {}
    
    # Get all unique keys from both predictions
    all_keys = set(prediction1.keys()) | set(prediction2.keys())
    
    for label in all_keys:
        val1 = prediction1.get(label, 0.0)
        val2 = prediction2.get(label, 0.0)
        
        difference = val2 - val1
        
        # Calculate percentage change (avoid division by zero)
        if val1 != 0:
            percent_change = (difference / val1) * 100
        elif val2 != 0:
            percent_change = 100.0  # If val1 is 0 but val2 is not, it's a 100% increase
        else:
            percent_change = 0.0  # Both are 0
        
        comparison[label] = {
            'first': val1,
            'second': val2,
            'difference': difference,
            'percent_change': percent_change
        }
    
    return comparison
