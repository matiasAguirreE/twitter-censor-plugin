"""
Sentiment Analysis Module using RoBERTuito
Provides sentiment analysis as a second layer to mitigate bias in toxicity detection
"""

import logging
from typing import Dict, Optional, Tuple
from pysentimiento import create_analyzer
from .device_utils import get_optimal_device

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Sentiment analyzer using RoBERTuito for Spanish text
    Provides sentiment classification to help correct toxicity detection bias
    """
    
    def __init__(self):
        self.analyzer = None
        self.device, self.device_name = get_optimal_device()
        self._load_analyzer()
    
    def _load_analyzer(self):
        """Load RoBERTuito sentiment analyzer"""
        try:
            logger.info("🤖 Loading RoBERTuito sentiment analyzer...")
            self.analyzer = create_analyzer(task="sentiment", lang="es")
            logger.info("✅ RoBERTuito sentiment analyzer loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading sentiment analyzer: {e}")
            self.analyzer = None
    
    def is_available(self) -> bool:
        """Check if sentiment analyzer is available"""
        return self.analyzer is not None
    
    def predict_sentiment(self, text: str) -> Optional[Dict]:
        """
        Predict sentiment for given text
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict with sentiment prediction or None if analyzer not available
        """
        if not self.is_available():
            logger.warning("⚠️ Sentiment analyzer not available")
            return None
        
        try:
            result = self.analyzer.predict(text)
            
            # Convert to our standard format
            sentiment_result = {
                'label': result.output,
                'probabilities': dict(result.probas),
                'confidence': max(result.probas.values())
            }
            
            return sentiment_result
            
        except Exception as e:
            logger.error(f"❌ Error predicting sentiment: {e}")
            return None

class CoherenceAnalyzer:
    """
    Analyzer for detecting inconsistencies between sentiment and toxicity without using word lists
    Uses statistical analysis and model coherence detection
    """
    
    # Statistical thresholds for detecting anomalies - OPTIMIZED FOR BETTER DETECTION
    HIGH_COHERENCE_MISMATCH_THRESHOLD = 0.5    # Reduced for more sensitive detection
    MODERATE_COHERENCE_MISMATCH_THRESHOLD = 0.3  # Reduced for broader detection
    ANOMALY_DETECTION_THRESHOLD = 0.6          # Significantly reduced for more anomaly detection
    
    @staticmethod
    def calculate_sentiment_toxicity_coherence(sentiment_result: Dict, toxicity_scores: Dict) -> Dict:
        """
        Calculate coherence between sentiment analysis and toxicity detection
        Returns statistical measures of coherence/mismatch
        """
        if not sentiment_result:
            return {'coherence_score': 0.0, 'mismatch_type': 'no_sentiment'}
        
        sentiment_label = sentiment_result['label']
        sentiment_confidence = sentiment_result['confidence']
        max_toxicity = max(toxicity_scores.values())
        avg_toxicity = sum(toxicity_scores.values()) / len(toxicity_scores)
        
        # Calculate coherence score based on expected relationships
        if sentiment_label == 'POS':
            # Positive sentiment should have low toxicity
            expected_toxicity = 0.1
            coherence_score = 1.0 - abs(max_toxicity - expected_toxicity)
        elif sentiment_label == 'NEG':
            # Negative sentiment could have higher toxicity, but not always
            # We'll be more lenient here to avoid false corrections
            expected_toxicity = min(0.4, max_toxicity)  # Allow some toxicity for negative sentiment
            coherence_score = 1.0 - max(0, max_toxicity - 0.7)  # Only penalize very high toxicity
        else:  # NEU
            # Neutral sentiment should have low-moderate toxicity
            expected_toxicity = 0.2
            coherence_score = 1.0 - max(0, max_toxicity - 0.5)  # Allow moderate toxicity for neutral
        
        # Detect mismatch types based on statistical analysis
        mismatch_type = "coherent"
        mismatch_severity = 0.0
        
        if sentiment_label == 'POS' and max_toxicity > 0.4:  # Reduced from 0.7 to 0.4
            mismatch_type = "positive_sentiment_high_toxicity"
            mismatch_severity = (max_toxicity - 0.4) * sentiment_confidence
        elif sentiment_label == 'NEU' and max_toxicity > 0.3:  # Reduced from 0.6 to 0.3
            mismatch_type = "neutral_sentiment_high_toxicity"
            mismatch_severity = (max_toxicity - 0.3) * sentiment_confidence
        elif sentiment_label == 'NEG' and max_toxicity > 0.95:  # Slightly reduced from 0.9 to 0.95
            # Only flag extreme cases for negative sentiment
            mismatch_type = "negative_sentiment_extreme_toxicity"
            mismatch_severity = (max_toxicity - 0.95) * sentiment_confidence
        
        return {
            'coherence_score': coherence_score,
            'mismatch_type': mismatch_type,
            'mismatch_severity': mismatch_severity,
            'sentiment_confidence': sentiment_confidence,
            'max_toxicity': max_toxicity,
            'avg_toxicity': avg_toxicity
        }
    
    @staticmethod
    def detect_statistical_anomaly(toxicity_scores: Dict, sentiment_result: Dict) -> Dict:
        """
        Detect statistical anomalies in the toxicity distribution
        """
        if not sentiment_result:
            return {'is_anomaly': False, 'anomaly_type': 'no_sentiment'}
        
        scores = list(toxicity_scores.values())
        max_score = max(scores)
        min_score = min(scores)
        score_range = max_score - min_score
        
        # Calculate score distribution statistics
        mean_score = sum(scores) / len(scores)
        variance = sum((x - mean_score) ** 2 for x in scores) / len(scores)
        std_dev = variance ** 0.5
        
        # Detect anomalies
        is_anomaly = False
        anomaly_type = "normal"
        anomaly_strength = 0.0
        
        # Anomaly 1: Single extremely high score with others very low - MORE SENSITIVE
        if max_score > 0.5 and score_range > 0.4 and std_dev > 0.2:  # Reduced thresholds
            is_anomaly = True
            anomaly_type = "single_extreme_outlier"
            anomaly_strength = max_score * (score_range / 1.0)
        
        # Anomaly 2: High toxicity with confident positive sentiment - MORE SENSITIVE
        elif (max_score > 0.4 and   # Reduced from 0.7 to 0.4
              sentiment_result['label'] == 'POS' and 
              sentiment_result['confidence'] > 0.6):  # Reduced from 0.8 to 0.6
            is_anomaly = True
            anomaly_type = "high_toxicity_confident_positive"
            anomaly_strength = max_score * sentiment_result['confidence']
        
        # Anomaly 3: Moderate toxicity with confident neutral sentiment - MORE SENSITIVE
        elif (max_score > 0.3 and   # Reduced from 0.6 to 0.3
              sentiment_result['label'] == 'NEU' and 
              sentiment_result['confidence'] > 0.6):  # Reduced from 0.7 to 0.6
            is_anomaly = True
            anomaly_type = "moderate_toxicity_confident_neutral"
            anomaly_strength = max_score * sentiment_result['confidence'] * 0.8
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_type': anomaly_type,
            'anomaly_strength': anomaly_strength,
            'score_distribution': {
                'mean': mean_score,
                'std_dev': std_dev,
                'range': score_range,
                'max': max_score,
                'min': min_score
            }
        }
    
    @staticmethod
    def should_apply_coherence_correction(toxicity_scores: Dict, sentiment_result: Dict) -> str:
        """
        Determine if coherence-based correction should be applied using statistical analysis
        Now includes minimum confidence threshold check
        
        Returns:
            str: Type of coherence correction to apply, or empty string if none
        """
        if not sentiment_result:
            return ""
        
        # NEW: Check minimum confidence threshold first (same as BiasCorrector)
        sentiment_confidence = sentiment_result['confidence']
        if sentiment_confidence < BiasCorrector.MINIMUM_CONFIDENCE_THRESHOLD:
            logger.info(f"🚫 Coherence analysis skipped - sentiment confidence too low ({sentiment_confidence:.1%})")
            return ""
        
        # Analyze coherence and anomalies
        coherence_analysis = CoherenceAnalyzer.calculate_sentiment_toxicity_coherence(
            sentiment_result, toxicity_scores
        )
        anomaly_analysis = CoherenceAnalyzer.detect_statistical_anomaly(
            toxicity_scores, sentiment_result
        )
        
        # Determine correction type based on analysis - MORE AGGRESSIVE
        if anomaly_analysis['is_anomaly']:
            if coherence_analysis['mismatch_severity'] > 0.1:  # Reduced from 0.5 to 0.1
                return anomaly_analysis['anomaly_type']
        
        if coherence_analysis['mismatch_type'] != "coherent":
            if coherence_analysis['mismatch_severity'] > 0.05:  # Reduced from 0.3 to 0.05
                return coherence_analysis['mismatch_type']
        
        return ""

class BiasCorrector:
    """
    Enhanced bias corrector with confidence thresholds and negativity scaling
    """
    
    # Enhanced thresholds and parameters
    HIGH_TOXICITY_THRESHOLD = 0.7
    POSITIVE_SENTIMENT_THRESHOLD = 0.6    
    NEUTRAL_SENTIMENT_THRESHOLD = 0.65    
    
    # NEW: Minimum confidence threshold for applying corrections
    MINIMUM_CONFIDENCE_THRESHOLD = 0.65   # Require 65% confidence to apply correction
    
    # Reduced correction factors (less aggressive)
    BASE_CORRECTION_FACTOR = 0.25          # Reduced from 40% to 25%
    CONFIDENCE_MULTIPLIER = 0.6            # Reduced from 1.0 to 0.6
    MAX_CORRECTION_FACTOR = 0.7            # Reduced from 90% to 70%
    MIN_TOXICITY_AFTER_CORRECTION = 0.1   # Increased from 5% to 10%
    
    # NEW: Negativity scaling parameters
    NEGATIVE_SENTIMENT_SCALING = True      # Enable scaling up for negative sentiment
    NEGATIVE_SCALING_FACTOR = 0.15         # 15% increase for confident negative sentiment
    NEGATIVE_CONFIDENCE_THRESHOLD = 0.65   # Require 65% confidence for scaling
    MAX_SCALING_LIMIT = 0.95               # Don't scale beyond 95%
    
    # Enhanced confidence thresholds
    VERY_HIGH_CONFIDENCE_THRESHOLD = 0.85
    VERY_HIGH_CONFIDENCE_BONUS = 0.2       # Reduced from 30% to 20%
    
    # Neutral correction
    NEUTRAL_CORRECTION_FACTOR = 0.3        # Reduced from 50% to 30%
    
    @staticmethod
    def should_apply_correction(toxicity_scores: Dict, sentiment_result: Dict) -> bool:
        """
        Enhanced logic to determine if bias correction should be applied
        Now includes minimum confidence threshold
        """
        if not sentiment_result:
            return False
        
        sentiment_label = sentiment_result['label']
        sentiment_confidence = sentiment_result['confidence']
        max_toxicity = max(toxicity_scores.values())
        
        # NEW: Check minimum confidence threshold
        if sentiment_confidence < BiasCorrector.MINIMUM_CONFIDENCE_THRESHOLD:
            logger.info(f"🚫 Sentiment confidence too low ({sentiment_confidence:.1%}) - skipping correction")
            return False
        
        # Original logic for high toxicity + positive sentiment
        if (sentiment_label == 'POS' and 
            sentiment_confidence > BiasCorrector.POSITIVE_SENTIMENT_THRESHOLD and
            max_toxicity > BiasCorrector.HIGH_TOXICITY_THRESHOLD):
            return True
        
        # Logic for neutral sentiment with demographic mentions + high toxicity
        if (sentiment_label == 'NEU' and 
            sentiment_confidence > BiasCorrector.NEUTRAL_SENTIMENT_THRESHOLD and
            max_toxicity > BiasCorrector.HIGH_TOXICITY_THRESHOLD):
            return True
        
        # Logic for extreme cases with any non-negative sentiment
        if (max_toxicity > 0.95 and 
            sentiment_label in ['POS', 'NEU'] and
            sentiment_confidence > BiasCorrector.MINIMUM_CONFIDENCE_THRESHOLD):
            return True
        
        return False
    
    @staticmethod
    def apply_correction(toxicity_scores: Dict, sentiment_result: Dict) -> Dict:
        """
        Enhanced correction that includes negativity scaling
        """
        corrected_scores = toxicity_scores.copy()
        
        if not sentiment_result:
            return corrected_scores
        
        sentiment_label = sentiment_result['label']
        sentiment_confidence = sentiment_result['confidence']
        
        # NEW: Apply negativity scaling if enabled and conditions are met
        if (BiasCorrector.NEGATIVE_SENTIMENT_SCALING and 
            sentiment_label == 'NEG' and 
            sentiment_confidence >= BiasCorrector.NEGATIVE_CONFIDENCE_THRESHOLD):
            
            # Scale up toxicity scores slightly for confident negative sentiment
            for label in corrected_scores:
                original_score = corrected_scores[label]
                scaled_score = original_score * (1 + BiasCorrector.NEGATIVE_SCALING_FACTOR)
                # Apply scaling limit
                corrected_scores[label] = min(scaled_score, BiasCorrector.MAX_SCALING_LIMIT)
            
            logger.info(f"📈 Applied negativity scaling (+{BiasCorrector.NEGATIVE_SCALING_FACTOR:.1%}) for confident negative sentiment")
            return corrected_scores
        
        # Apply standard bias correction if conditions are met
        if BiasCorrector.should_apply_correction(toxicity_scores, sentiment_result):
            correction_factor = BiasCorrector.calculate_correction_factor(sentiment_result)
            
            for label in corrected_scores:
                original_score = corrected_scores[label]
                reduced_score = original_score * (1 - correction_factor)
                corrected_scores[label] = max(reduced_score, BiasCorrector.MIN_TOXICITY_AFTER_CORRECTION)
            
            logger.info(f"🔧 Applied bias correction (factor: {correction_factor:.1%}) for {sentiment_label} sentiment")
        
        return corrected_scores
    
    @staticmethod
    def calculate_correction_factor(sentiment_result: Dict) -> float:
        """
        Calculate the adaptive correction factor with reduced aggressiveness
        """
        if not sentiment_result:
            return 0.0
        
        sentiment_label = sentiment_result['label']
        sentiment_confidence = sentiment_result['confidence']
        
        # Check minimum confidence threshold first
        if sentiment_confidence < BiasCorrector.MINIMUM_CONFIDENCE_THRESHOLD:
            return 0.0
        
        # Calculate base correction factor
        if sentiment_label == 'POS' and sentiment_confidence > BiasCorrector.POSITIVE_SENTIMENT_THRESHOLD:
            # Base correction
            base_factor = BiasCorrector.BASE_CORRECTION_FACTOR
            
            # Confidence bonus (reduced)
            confidence_bonus = (sentiment_confidence - BiasCorrector.POSITIVE_SENTIMENT_THRESHOLD) * BiasCorrector.CONFIDENCE_MULTIPLIER
            
            # Very high confidence bonus (reduced)
            high_confidence_bonus = 0.0
            if sentiment_confidence > BiasCorrector.VERY_HIGH_CONFIDENCE_THRESHOLD:
                high_confidence_bonus = BiasCorrector.VERY_HIGH_CONFIDENCE_BONUS
            
            # Calculate total factor with maximum limit
            total_factor = min(
                base_factor + confidence_bonus + high_confidence_bonus,
                BiasCorrector.MAX_CORRECTION_FACTOR
            )
            
            return total_factor
            
        elif sentiment_label == 'NEU' and sentiment_confidence > BiasCorrector.NEUTRAL_SENTIMENT_THRESHOLD:
            # Neutral sentiment correction (reduced)
            return BiasCorrector.NEUTRAL_CORRECTION_FACTOR
        
        return 0.0
    
    @staticmethod
    def get_bias_flags(toxicity_scores: Dict, sentiment_result: Dict) -> Dict:
        """
        Enhanced bias detection flags with confidence thresholds
        """
        if not sentiment_result:
            return {
                'potential_bias_detected': False,
                'high_toxicity_positive_sentiment': False,
                'sentiment_toxicity_mismatch': False,
                'correction_applied': False,
                'confidence_too_low': False
            }
        
        sentiment_label = sentiment_result['label']
        sentiment_confidence = sentiment_result['confidence']
        max_toxicity = max(toxicity_scores.values())
        
        # Check confidence threshold
        confidence_too_low = sentiment_confidence < BiasCorrector.MINIMUM_CONFIDENCE_THRESHOLD
        
        # Enhanced bias detection
        high_toxicity_positive_sentiment = (
            sentiment_label == 'POS' and 
            sentiment_confidence > BiasCorrector.POSITIVE_SENTIMENT_THRESHOLD and
            max_toxicity > BiasCorrector.HIGH_TOXICITY_THRESHOLD
        )
        
        sentiment_toxicity_mismatch = (
            (sentiment_label == 'POS' and max_toxicity > 0.5) or
            (sentiment_label == 'NEU' and max_toxicity > 0.6)
        )
        
        potential_bias_detected = (
            high_toxicity_positive_sentiment or 
            (sentiment_label == 'NEU' and max_toxicity > BiasCorrector.HIGH_TOXICITY_THRESHOLD)
        ) and not confidence_too_low
        
        correction_applied = BiasCorrector.should_apply_correction(toxicity_scores, sentiment_result)
        
        # Check for negativity scaling
        negativity_scaled = (
            BiasCorrector.NEGATIVE_SENTIMENT_SCALING and
            sentiment_label == 'NEG' and
            sentiment_confidence >= BiasCorrector.NEGATIVE_CONFIDENCE_THRESHOLD
        )
        
        return {
            'potential_bias_detected': potential_bias_detected,
            'high_toxicity_positive_sentiment': high_toxicity_positive_sentiment,
            'sentiment_toxicity_mismatch': sentiment_toxicity_mismatch,
            'correction_applied': correction_applied,
            'confidence_too_low': confidence_too_low,
            'negativity_scaled': negativity_scaled,
            'sentiment_confidence': sentiment_confidence,
            'min_confidence_threshold': BiasCorrector.MINIMUM_CONFIDENCE_THRESHOLD
        }

# Global sentiment analyzer instance
_sentiment_analyzer = None

def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get global sentiment analyzer instance (singleton pattern)"""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer

def analyze_with_sentiment_correction(text: str, toxicity_scores: Dict) -> Dict:
    """
    Complete analysis with sentiment-based bias correction and coherence analysis
    
    Args:
        text: Input text
        toxicity_scores: Original toxicity prediction scores
        
    Returns:
        Dict with complete analysis including sentiment and bias correction
    """
    analyzer = get_sentiment_analyzer()
    
    # Get sentiment analysis
    sentiment_result = analyzer.predict_sentiment(text)
    
    # Perform coherence analysis (no word lists needed)
    coherence_type = CoherenceAnalyzer.should_apply_coherence_correction(toxicity_scores, sentiment_result)
    
    # Apply coherence-based corrections first if detected
    if coherence_type:
        corrected_scores = apply_coherence_correction(toxicity_scores, sentiment_result, coherence_type)
        coherence_correction_applied = True
    else:
        # Apply standard bias correction
        corrected_scores = BiasCorrector.apply_correction(toxicity_scores, sentiment_result)
        coherence_correction_applied = False
    
    # Calculate correction factor (for transparency)
    correction_factor = BiasCorrector.calculate_correction_factor(sentiment_result)
    
    # Get bias flags
    bias_flags = BiasCorrector.get_bias_flags(toxicity_scores, sentiment_result)
    
    # Get detailed coherence analysis
    coherence_analysis = CoherenceAnalyzer.calculate_sentiment_toxicity_coherence(
        sentiment_result, toxicity_scores
    )
    anomaly_analysis = CoherenceAnalyzer.detect_statistical_anomaly(
        toxicity_scores, sentiment_result
    )
    
    # Add coherence analysis flags
    coherence_flags = {
        'coherence_score': coherence_analysis.get('coherence_score', 0.0),
        'mismatch_type': coherence_analysis.get('mismatch_type', 'unknown'),
        'mismatch_severity': coherence_analysis.get('mismatch_severity', 0.0),
        'is_statistical_anomaly': anomaly_analysis.get('is_anomaly', False),
        'anomaly_type': anomaly_analysis.get('anomaly_type', 'normal'),
        'anomaly_strength': anomaly_analysis.get('anomaly_strength', 0.0),
        'score_distribution': anomaly_analysis.get('score_distribution', {}),
        'coherence_correction_type': coherence_type,
        'coherence_correction_applied': coherence_correction_applied
    }
    
    return {
        'original_toxicity': toxicity_scores,
        'corrected_toxicity': corrected_scores,
        'sentiment_analysis': sentiment_result,
        'bias_analysis': bias_flags,
        'coherence_analysis': coherence_flags,
        'correction_applied': bias_flags['correction_applied'] or coherence_correction_applied,
        'correction_factor': correction_factor,
        'correction_details': {
            'base_factor': BiasCorrector.BASE_CORRECTION_FACTOR,
            'confidence_bonus': max(0, (sentiment_result['confidence'] - BiasCorrector.POSITIVE_SENTIMENT_THRESHOLD) * BiasCorrector.CONFIDENCE_MULTIPLIER) if sentiment_result and sentiment_result['label'] == 'POS' else 0,
            'total_factor': correction_factor,
            'max_possible_factor': BiasCorrector.MAX_CORRECTION_FACTOR,
            'coherence_type': coherence_type
        }
    }

def apply_coherence_correction(toxicity_scores: Dict, sentiment_result: Dict, coherence_type: str) -> Dict:
    """
    Apply coherence-based corrections using statistical analysis instead of word patterns
    
    Args:
        toxicity_scores: Original toxicity scores
        sentiment_result: Sentiment analysis results
        coherence_type: Type of coherence issue detected
        
    Returns:
        Dict with corrected toxicity scores
    """
    corrected_scores = toxicity_scores.copy()
    
    # Get detailed analysis for adaptive correction
    coherence_analysis = CoherenceAnalyzer.calculate_sentiment_toxicity_coherence(
        sentiment_result, toxicity_scores
    )
    anomaly_analysis = CoherenceAnalyzer.detect_statistical_anomaly(
        toxicity_scores, sentiment_result
    )
    
    # Calculate adaptive correction factors based on statistical measures
    max_toxicity = max(toxicity_scores.values())
    sentiment_confidence = sentiment_result['confidence']
    mismatch_severity = coherence_analysis.get('mismatch_severity', 0.0)
    anomaly_strength = anomaly_analysis.get('anomaly_strength', 0.0)
    
    if coherence_type == "positive_sentiment_high_toxicity":
        # Strong correction for positive sentiment with high toxicity
        correction_factor = 0.7 + (sentiment_confidence - 0.6) * 0.5  # 70-90% reduction
        correction_factor = min(correction_factor, 0.95)
        
        # Extra aggressive for demographic categories with high mismatch
        for label in ['Xenofobia', 'Homofobia']:
            if corrected_scores[label] > 0.5:
                enhanced_factor = min(correction_factor * 1.2, 0.95)
                corrected_scores[label] = max(corrected_scores[label] * (1 - enhanced_factor), 0.05)
        
        # Standard correction for other categories
        for label in ['Violencia']:
            if corrected_scores[label] > 0.5:
                corrected_scores[label] = max(corrected_scores[label] * (1 - correction_factor), 0.05)
        
        logger.info(f"🔧 Applied positive sentiment coherence correction (factor: {correction_factor:.3f})")
    
    elif coherence_type == "neutral_sentiment_high_toxicity":
        # Moderate correction for neutral sentiment with high toxicity
        correction_factor = 0.4 + (sentiment_confidence - 0.6) * 0.3  # 40-70% reduction
        correction_factor = min(correction_factor, 0.8)
        
        # Focus on demographic categories for neutral sentiment
        for label in ['Xenofobia', 'Homofobia']:
            if corrected_scores[label] > 0.4:
                corrected_scores[label] = max(corrected_scores[label] * (1 - correction_factor), 0.05)
        
        logger.info(f"🔧 Applied neutral sentiment coherence correction (factor: {correction_factor:.3f})")
    
    elif coherence_type == "single_extreme_outlier":
        # Correction for statistical anomaly - single extreme score
        score_distribution = anomaly_analysis.get('score_distribution', {})
        max_score = score_distribution.get('max', 0)
        std_dev = score_distribution.get('std_dev', 0)
        
        # Find the outlier category and correct it
        for label, score in corrected_scores.items():
            if score == max_score and score > 0.7:
                # Strong correction for the outlier
                correction_factor = 0.6 + min(std_dev, 0.3)  # 60-90% reduction
                corrected_scores[label] = max(score * (1 - correction_factor), 0.05)
                logger.info(f"🔧 Applied outlier correction to {label} (factor: {correction_factor:.3f})")
                break
    
    elif coherence_type == "high_toxicity_confident_positive":
        # Very aggressive correction for high confidence positive with high toxicity
        correction_factor = 0.8 + (sentiment_confidence - 0.8) * 0.15  # 80-95% reduction
        correction_factor = min(correction_factor, 0.95)
        
        # Apply to all categories proportionally
        for label in corrected_scores:
            if corrected_scores[label] > 0.3:
                corrected_scores[label] = max(corrected_scores[label] * (1 - correction_factor), 0.05)
        
        logger.info(f"🔧 Applied high confidence positive correction (factor: {correction_factor:.3f})")
    
    elif coherence_type == "moderate_toxicity_confident_neutral":
        # Targeted correction for confident neutral with moderate toxicity  
        correction_factor = 0.5 + (sentiment_confidence - 0.7) * 0.2  # 50-70% reduction
        correction_factor = min(correction_factor, 0.8)
        
        # Focus on demographic categories
        for label in ['Xenofobia', 'Homofobia']:
            if corrected_scores[label] > 0.3:
                corrected_scores[label] = max(corrected_scores[label] * (1 - correction_factor), 0.05)
        
        logger.info(f"🔧 Applied confident neutral correction (factor: {correction_factor:.3f})")
    
    elif coherence_type == "negative_sentiment_extreme_toxicity":
        # Conservative correction for negative sentiment with extreme toxicity
        # Only correct if toxicity is truly extreme (>95%) to avoid over-correction
        correction_factor = 0.2  # Only 20% reduction for negative sentiment
        
        for label in corrected_scores:
            if corrected_scores[label] > 0.95:
                corrected_scores[label] = max(corrected_scores[label] * (1 - correction_factor), 0.1)
        
        logger.info(f"🔧 Applied conservative negative sentiment correction (factor: {correction_factor:.3f})")
    
    return corrected_scores

if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    if analyzer.is_available():
        test_texts = [
            "Ayer salí a bailar con mi mejor amiga que es venezolana",
            "Creo que los homosexuales aportan cultura y valor al país",
            "Los venezolanos son una mierda",
            "Odio a los homosexuales"
        ]
        
        for text in test_texts:
            print(f"\nTexto: {text}")
            sentiment = analyzer.predict_sentiment(text)
            if sentiment:
                print(f"Sentimiento: {sentiment['label']} (confianza: {sentiment['confidence']:.3f})")
                print(f"Probabilidades: {sentiment['probabilities']}")
    else:
        print("❌ Sentiment analyzer not available") 