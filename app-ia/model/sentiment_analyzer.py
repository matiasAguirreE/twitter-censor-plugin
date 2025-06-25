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
        
        Returns:
            str: Type of coherence correction to apply, or empty string if none
        """
        if not sentiment_result:
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
    Bias correction logic that combines toxicity detection with sentiment analysis
    """
    
    # Thresholds for bias correction
    HIGH_TOXICITY_THRESHOLD = 0.7
    POSITIVE_SENTIMENT_THRESHOLD = 0.6
    NEUTRAL_SENTIMENT_THRESHOLD = 0.65  # For neutral sentiment corrections
    
    # Adaptive correction parameters
    BASE_CORRECTION_FACTOR = 0.4      # Increased base reduction (40%)
    CONFIDENCE_MULTIPLIER = 1.0       # Increased multiplier for stronger corrections
    MAX_CORRECTION_FACTOR = 0.9       # Increased maximum total reduction (90%)
    MIN_TOXICITY_AFTER_CORRECTION = 0.05  # Lower minimum for stronger corrections
    
    # Neutral sentiment correction (for demographic mentions in neutral context)
    NEUTRAL_CORRECTION_FACTOR = 0.5   # 50% reduction for neutral demographic mentions
    
    # Enhanced correction for very high confidence positive sentiment
    VERY_HIGH_CONFIDENCE_THRESHOLD = 0.85
    VERY_HIGH_CONFIDENCE_BONUS = 0.3  # Additional 30% reduction for very confident positive
    
    @staticmethod
    def should_apply_correction(toxicity_scores: Dict, sentiment_result: Dict) -> bool:
        """
        Determine if bias correction should be applied
        
        Args:
            toxicity_scores: Dict with toxicity class probabilities
            sentiment_result: Dict with sentiment analysis results
            
        Returns:
            bool: True if correction should be applied
        """
        if not sentiment_result:
            return False
        
        # Check if any toxicity score is high
        max_toxicity = max(toxicity_scores.values())
        has_high_toxicity = max_toxicity > BiasCorrector.HIGH_TOXICITY_THRESHOLD
        
        # Check for different correction scenarios
        sentiment_label = sentiment_result['label']
        sentiment_confidence = sentiment_result['confidence']
        
        # Scenario 1: Positive sentiment (original logic)
        is_positive_sentiment = (
            sentiment_label == 'POS' and 
            sentiment_confidence > BiasCorrector.POSITIVE_SENTIMENT_THRESHOLD
        )
        
        # Scenario 2: Neutral sentiment with high demographic toxicity
        is_neutral_demographic = (
            sentiment_label == 'NEU' and 
            sentiment_confidence > BiasCorrector.NEUTRAL_SENTIMENT_THRESHOLD and
            (toxicity_scores.get('Xenofobia', 0) > 0.5 or toxicity_scores.get('Homofobia', 0) > 0.3)
        )
        
        # Scenario 3: Very high toxicity with any non-negative sentiment
        is_extreme_toxicity = (
            max_toxicity > 0.9 and 
            sentiment_label != 'NEG'
        )
        
        return has_high_toxicity and (is_positive_sentiment or is_neutral_demographic or is_extreme_toxicity)
    
    @staticmethod
    def apply_correction(toxicity_scores: Dict, sentiment_result: Dict) -> Dict:
        """
        Apply adaptive bias correction to toxicity scores based on sentiment confidence
        
        Args:
            toxicity_scores: Original toxicity scores
            sentiment_result: Sentiment analysis results
            
        Returns:
            Dict with corrected toxicity scores
        """
        corrected_scores = toxicity_scores.copy()
        
        if BiasCorrector.should_apply_correction(toxicity_scores, sentiment_result):
            sentiment_label = sentiment_result['label']
            sentiment_confidence = sentiment_result['confidence']
            max_toxicity = max(toxicity_scores.values())
            
            # Determine correction type and factor
            if sentiment_label == 'POS':
                # Positive sentiment correction (enhanced)
                confidence_bonus = (sentiment_confidence - BiasCorrector.POSITIVE_SENTIMENT_THRESHOLD) * BiasCorrector.CONFIDENCE_MULTIPLIER
                
                # Extra bonus for very high confidence positive sentiment
                if sentiment_confidence > BiasCorrector.VERY_HIGH_CONFIDENCE_THRESHOLD:
                    confidence_bonus += BiasCorrector.VERY_HIGH_CONFIDENCE_BONUS
                
                # Extra aggressive correction for extreme toxicity + positive sentiment
                if max_toxicity > 0.95 and sentiment_confidence > 0.8:
                    confidence_bonus += 0.2  # Additional 20% for extreme cases
                
                total_correction_factor = min(
                    BiasCorrector.BASE_CORRECTION_FACTOR + confidence_bonus,
                    BiasCorrector.MAX_CORRECTION_FACTOR
                )
                correction_type = "positive_sentiment"
                
            elif sentiment_label == 'NEU':
                # Neutral sentiment with demographic mention
                total_correction_factor = BiasCorrector.NEUTRAL_CORRECTION_FACTOR
                correction_type = "neutral_demographic"
                
            else:
                # Extreme toxicity with non-negative sentiment
                total_correction_factor = BiasCorrector.BASE_CORRECTION_FACTOR * 0.7  # More conservative
                correction_type = "extreme_toxicity"
            
            # Apply correction to each toxicity score
            for label in corrected_scores:
                original_score = corrected_scores[label]
                
                # Special handling for demographic categories
                if label in ['Xenofobia', 'Homofobia'] and sentiment_label in ['POS', 'NEU']:
                    # More aggressive correction for demographic bias
                    enhanced_factor = min(total_correction_factor * 1.2, 0.95)
                    corrected_score = original_score * (1 - enhanced_factor)
                else:
                    # Standard correction
                    corrected_score = original_score * (1 - total_correction_factor)
                
                # Ensure minimum toxicity threshold
                corrected_scores[label] = max(corrected_score, BiasCorrector.MIN_TOXICITY_AFTER_CORRECTION)
            
            logger.info(f"🔧 Applied {correction_type} correction - sentiment: {sentiment_label} "
                       f"(confidence: {sentiment_confidence:.3f}, correction factor: {total_correction_factor:.3f})")
        
        return corrected_scores
    
    @staticmethod
    def calculate_correction_factor(sentiment_result: Dict) -> float:
        """
        Calculate the correction factor that would be applied
        
        Args:
            sentiment_result: Sentiment analysis results
            
        Returns:
            float: Correction factor (0.0 to MAX_CORRECTION_FACTOR)
        """
        if not sentiment_result or sentiment_result['label'] != 'POS':
            return 0.0
        
        sentiment_confidence = sentiment_result['confidence']
        
        if sentiment_confidence <= BiasCorrector.POSITIVE_SENTIMENT_THRESHOLD:
            return 0.0
        
        # Calculate adaptive correction factor
        confidence_bonus = (sentiment_confidence - BiasCorrector.POSITIVE_SENTIMENT_THRESHOLD) * BiasCorrector.CONFIDENCE_MULTIPLIER
        total_correction_factor = min(
            BiasCorrector.BASE_CORRECTION_FACTOR + confidence_bonus,
            BiasCorrector.MAX_CORRECTION_FACTOR
        )
        
        return total_correction_factor
    
    @staticmethod
    def get_bias_flags(toxicity_scores: Dict, sentiment_result: Dict) -> Dict:
        """
        Get bias detection flags for analysis
        
        Args:
            toxicity_scores: Toxicity prediction scores
            sentiment_result: Sentiment analysis results
            
        Returns:
            Dict with bias analysis flags
        """
        flags = {
            'potential_bias_detected': False,
            'high_toxicity_positive_sentiment': False,
            'sentiment_toxicity_mismatch': False,
            'correction_applied': False
        }
        
        if not sentiment_result:
            return flags
        
        max_toxicity = max(toxicity_scores.values())
        
        # Check for potential bias patterns
        if max_toxicity > BiasCorrector.HIGH_TOXICITY_THRESHOLD:
            if sentiment_result['label'] == 'POS':
                flags['high_toxicity_positive_sentiment'] = True
                flags['potential_bias_detected'] = True
            elif sentiment_result['label'] == 'NEU' and sentiment_result['confidence'] > 0.5:
                flags['sentiment_toxicity_mismatch'] = True
                flags['potential_bias_detected'] = True
        
        # Check if correction was applied
        flags['correction_applied'] = BiasCorrector.should_apply_correction(toxicity_scores, sentiment_result)
        
        return flags

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