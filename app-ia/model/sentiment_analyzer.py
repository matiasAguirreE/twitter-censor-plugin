import logging
from typing import Dict, Optional
from pysentimiento import create_analyzer
from .device_utils import get_optimal_device

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Main Sentiment Analyzer Class ---
class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = None
        self.device, self.device_name = get_optimal_device()
        self._load_analyzer()

    def _load_analyzer(self):
        try:
            logger.info("🤖 Loading RoBERTuito sentiment analyzer...")
            self.analyzer = create_analyzer(task="sentiment", lang="es")
            logger.info("✅ RoBERTuito sentiment analyzer loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading sentiment analyzer: {e}")
            self.analyzer = None

    def is_available(self) -> bool:
        return self.analyzer is not None

    def predict_sentiment(self, text: str) -> Optional[Dict]:
        if not self.is_available():
            logger.warning("⚠️ Sentiment analyzer not available")
            return None
        try:
            result = self.analyzer.predict(text)
            return {
                'label': result.output,
                'confidence': max(result.probas.values())
            }
        except Exception as e:
            logger.error(f"❌ Error predicting sentiment: {e}")
            return None

# --- Rule-Based Correction Logic ---
def apply_correction_rules(toxicity_scores: Dict, sentiment_result: Dict) -> Dict:
    """
    Applies a set of specific, rule-based corrections to toxicity scores.
    """
    if not sentiment_result:
        return toxicity_scores.copy()

    corrected_scores = toxicity_scores.copy()
    sentiment_label = sentiment_result['label']
    sentiment_confidence = sentiment_result['confidence']

    # Rule #1: Corrects "False Positives by Intensity"
    # If sentiment is very positive and violence is high, it's likely a joke/metaphor.
    if (sentiment_label == 'POS' and
        sentiment_confidence > 0.90 and
        corrected_scores.get('Violencia', 0) > 0.70):
        
        original_score = corrected_scores['Violencia']
        # Drastically reduce the violence score
        corrected_scores['Violencia'] *= 0.1
        logger.info(f"✅ RULE 1 APPLIED: Corrected 'Violencia' from {original_score:.2f} to {corrected_scores['Violencia']:.2f} due to high positive sentiment.")

    # Rule #2: Corrects "False Negatives by Sutlety" (Experimental)
    # If sentiment is very negative and toxicity is in a "gray area", give it a push.
    elif (sentiment_label == 'NEG' and
          sentiment_confidence > 0.75):
          
        for label in ['Xenofobia', 'Homofobia']:
            score = corrected_scores.get(label, 0)
            if 0.2 < score < 0.6:
                original_score = score
                # Give a small push to cross the threshold
                corrected_scores[label] *= 1.8
                logger.info(f"✅ RULE 2 APPLIED: Boosted '{label}' from {original_score:.2f} to {corrected_scores[label]:.2f} due to high negative sentiment.")

    # Rule #3: Ignore Neutral Sentiment is implicitly handled by having no 'NEU' condition.

    return corrected_scores


# --- Main public functions ---

_sentiment_analyzer = None

def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get global sentiment analyzer instance (singleton pattern)"""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer

def analyze_with_sentiment_correction(text: str, toxicity_scores: Dict) -> Dict:
    """
    Refactored main analysis function using the new rule-based system.
    """
    analyzer = get_sentiment_analyzer()
    sentiment_result = analyzer.predict_sentiment(text)
    
    # Apply the new rule-based correction
    corrected_scores = apply_correction_rules(toxicity_scores, sentiment_result)
    
    correction_applied = (corrected_scores != toxicity_scores)

    return {
        'original_toxicity': toxicity_scores,
        'corrected_toxicity': corrected_scores,
        'sentiment_analysis': sentiment_result,
        'correction_applied': correction_applied,
        # Keep some fields for compatibility with evaluation script
        'bias_analysis': {},
        'coherence_analysis': {},
        'correction_factor': 0.0,
        'correction_details': {}
    }

if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    if analyzer.is_available():
        test_cases = {
            "Rule 1 (Hit)": {
                "text": "te amo tanto que te mato de amor",
                "scores": {"Violencia": 0.8, "Homofobia": 0.1, "Xenofobia": 0.1}
            },
            "Rule 2 (Hit)": {
                "text": "estos inmigrantes son una plaga, los detesto",
                "scores": {"Violencia": 0.2, "Homofobia": 0.1, "Xenofobia": 0.45}
            },
            "Rule 3 (Ignore)": {
                "text": "Aunque mi jefe es haitiano, es buen profesional.",
                "scores": {"Violencia": 0.1, "Homofobia": 0.1, "Xenofobia": 0.8}
            }
        }
        
        for name, case in test_cases.items():
            print(f"\n--- Testing Case: {name} ---")
            print(f"Text: {case['text']}")
            print(f"Original Scores: {case['scores']}")
            analysis = analyze_with_sentiment_correction(case['text'], case['scores'])
            print(f"Corrected Scores: {analysis['corrected_toxicity']}")
            print(f"Sentiment: {analysis['sentiment_analysis']}")
            print(f"Correction Applied: {analysis['correction_applied']}")
    else:
        print("❌ Sentiment analyzer not available")