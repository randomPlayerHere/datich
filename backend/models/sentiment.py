import os
from typing import Dict
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

class SentimentAnalyzer:
    """
    Wrapper class for your SLM model.
    Replace the model loading with your specific model.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name or local path
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name or os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        print(f"Loading model: {self.model_name} on {self.device}")
        
        # Option 1: Use HuggingFace pipeline (simpler)
        self.pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            device=0 if self.device == "cuda" else -1,
            top_k=None  # Return all scores
        )
        
        # Option 2: Load model directly (more control)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        # self.model.to(self.device)
        
        print("Model loaded successfully!")
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze text and return sentiment metrics.
        
        This is where you implement your custom logic to convert
        model outputs to dynamic metrics with labels.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with metrics array containing label/value pairs
        """
        # Get raw model output
        result = self.pipeline(text)
        
        # Example: Convert model output to metrics
        # This logic depends on your specific model's output format
        
        positive_score = 0
        negative_score = 0
        
        for item in result[0]:
            if item['label'].lower() in ['positive', 'pos', 'label_1']:
                positive_score = item['score']
            elif item['label'].lower() in ['negative', 'neg', 'label_0']:
                negative_score = item['score']
        
        # Derive metrics (replace with your model's logic)
        # Labels can be dynamic based on your model or text content
        anxiety = int(negative_score * 100 * 0.8)
        mood = int(positive_score * 100)
        stress = int((negative_score * 0.6 + (1 - positive_score) * 0.4) * 100)
        
        return {
            "metrics": [
                {"label": "Anxiety Indicators", "value": min(max(anxiety, 0), 100)},
                {"label": "Mood Stability", "value": min(max(mood, 0), 100)},
                {"label": "Stress Level", "value": min(max(stress, 0), 100)}
            ]
        }

# Singleton instance
_analyzer = None

def get_analyzer() -> SentimentAnalyzer:
    """Get or create the sentiment analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer