import re
import spacy
from typing import Dict, List, Set
import sys
sys.path.append('..')
from utils.config import config
from utils.logger import logger

class QueryParser:
    """
    Parse natural language queries using DistilBERT Question Answering
    Extracts attributes (colors, garments, scenes) using semantic understanding
    """
    
    def __init__(self):
        # Load DistilBERT QA pipeline
        try:
            from transformers import pipeline
            logger.info("Loading DistilBERT model for smart attribute extraction...")
            self.qa_model = pipeline(
                "question-answering", 
                model="distilbert-base-cased-distilled-squad",
                device=-1  # Use CPU
            )
        except Exception as e:
            logger.error(f"Error loading Transformers: {e}")
            logger.error("Please install via: pip install transformers torch")
            sys.exit(1)
            
        # Questions to ask the model for each attribute
        self.questions = {
            'colors': ["What is the color?", "What color is the clothing?"],
            'garments': ["What is the garment?", "What is the clothing item?", "What is the person wearing?"],
            'scenes': ["Where is the person?", "What is the location?", "What is the background?"],
            'style': ["What is the style?", "Is it formal or casual?"]
        }
        
        # Detection thresholds
        self.threshold = 0.2
    
    def parse_query(self, query: str) -> Dict:
        """
        Parse query using semantic QA model
        """
        # Store results
        extracted = {
            'colors': [],
            'garments': [],
            'scenes': [],
            'style': 'general'
        }
        
        # 1. Run QA for each attribute types
        for attr, questions in self.questions.items():
            best_score = 0
            best_answer = None
            
            # Try multiple phrasings of the question for robustness
            for q in questions:
                try:
                    # Get top 2 answers to handle "red shirt and blue pants"
                    results = self.qa_model(
                        question=q, 
                        context=query,
                        top_k=3,
                        handle_impossible_answer=True 
                    )
                    
                    # Normalize to list
                    if not isinstance(results, list):
                        results = [results]
                        
                    for res in results:
                        score = res['score']
                        ans = res['answer'].lower().strip()
                        
                        # Filter bad answers
                        if score > self.threshold and ans and ans != query.lower():
                            if attr == 'style':
                                # Only keep if it's a known style (optional constraint)
                                extracted['style'] = ans
                            elif ans not in extracted[attr]:
                                extracted[attr].append(ans)
                                
                except Exception as e:
                    continue
                    
        # 2. Check compositionality
        is_compositional = self._check_compositional(extracted)
        
        # 3. Suggest weights
        suggested_weights = self._suggest_weights(extracted, is_compositional)
        
        return {
            'raw_query': query,
            'colors': extracted['colors'],
            'garments': extracted['garments'],
            'scenes': extracted['scenes'],
            'style': extracted['style'],
            'is_compositional': is_compositional,
            'suggested_weights': suggested_weights
        }

    def _check_compositional(self, extracted: Dict) -> bool:
        """Check if query involves multiple specific attributes"""
        num_attributes = (len(extracted['colors']) + 
                         len(extracted['garments']) + 
                         len(extracted['scenes']))
        return num_attributes >= 2
    
    def _suggest_weights(self, extracted: Dict, is_compositional: bool) -> Dict:
        """
        Suggest adaptive weights based on query characteristics
        """
        weights = config.DEFAULT_WEIGHTS.copy()
        
        colors = extracted['colors']
        garments = extracted['garments']
        scenes = extracted['scenes']
        
        # Baseline boosts
        if colors: weights['color'] = 0.35
        if garments: weights['garment'] = 0.30
        if scenes: weights['scene'] = 0.30
        
        # Compositional logic
        if is_compositional:
            weights['clip'] = 0.25
            remaining = 0.75
            n_active = sum([1 for x in [colors, garments, scenes] if x])
            if n_active > 0:
                per_attribute = remaining / n_active
                weights['color'] = per_attribute if colors else 0.0
                weights['garment'] = per_attribute if garments else 0.0
                weights['scene'] = per_attribute if scenes else 0.0
        
        # Normalize
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

# Test it
if __name__ == "__main__":
    parser = QueryParser()
    
    test_queries = [
        "A person in a bright yellow raincoat",
        "Professional business attire inside a modern office",
        "Someone wearing a blue shirt sitting on a park bench",
        "Casual weekend outfit for a city walk",
        "A red tie and a white shirt in a formal setting"
    ]
    
    logger.info("\n" + "="*60)
    logger.info("Testing DistilBERT Query Parser")
    logger.info("="*60)
    
    for query in test_queries:
        logger.info(f"\n📝 Query: {query}")
        parsed = parser.parse_query(query)
        logger.info(f"   Colors: {parsed['colors']}")
        logger.info(f"   Garments: {parsed['garments']}")
        logger.info(f"   Scenes: {parsed['scenes']}")
        logger.info(f"   Style: {parsed['style']}")
        logger.info(f"   Compositional: {parsed['is_compositional']}")
        logger.info(f"   Suggested weights: {parsed['suggested_weights']}")