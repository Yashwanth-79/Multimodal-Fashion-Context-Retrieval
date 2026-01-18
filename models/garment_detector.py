import torch
import numpy as np
from typing import Union, List, Dict
from PIL import Image
import clip

class GarmentDetector:
    """
    Detect garment types using CLIP zero-shot classification
    Fashion-specific attribute extraction
    """
    
    def __init__(self, clip_encoder=None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use existing CLIP encoder or create new one
        if clip_encoder is None:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        else:
            self.model = clip_encoder.model
            self.preprocess = clip_encoder.preprocess
        self.model.eval()
        
        # Fashion-specific categories (hierarchical)
        self.garment_categories = self.garment_categories = {
                    'formal': [
                        'blazer', 'sport coat', 'suit jacket', 'suit',
                        'button-down shirt', 'button up', 'dress shirt',
                        'formal trousers', 'dress pants', 'slacks',
                        'pencil skirt', 'formal skirt',
                        'formal dress', 'evening dress', 'cocktail dress', 'gown',
                        'tie', 'necktie', 'bow tie', 'belt',
                        'formal shoes', 'dress shoes', 'oxfords',
                        'loafers', 'heels', 'pumps'
                    ],

                
                    'casual': [
                        't-shirt', 'tee', 'graphic tee',
                        'shirt', 'casual shirt',
                        'polo', 'tank top', 'crop top',
                        'hoodie', 'pullover', 'sweater', 'knitwear', 'sweatshirt',
                        'jeans', 'denims',
                        'joggers', 'track pants', 'shorts',
                        'leggings', 'chinos',
                        'casual dress', 'day dress', 'sundress',
                        'sneakers', 'trainers',
                        'flats', 'sandals', 'slippers',
                        'cap', 'baseball cap', 'hat',
                        'backpack', 'rucksack'
                    ],

                
                    'outerwear': [
                        'jacket', 'light jacket',
                        'coat', 'long coat',
                        'raincoat', 'waterproof jacket',
                        'parka', 'puffer jacket', 'down jacket',
                        'windbreaker',
                        'cardigan', 'knitted cardigan',
                        'vest', 'gilet',
                        'overcoat', 'trench coat'
                    ],

                
                    'activewear': [
                        'activewear', 'athletic wear', 'sportswear',
                        'gym clothes', 'workout clothes',
                        'yoga pants', 'leggings',
                        'running shorts', 'training shorts',
                        'athletic shoes', 'running shoes', 'sports shoes',
                        'trainers'
                    ],

                
                    'accessories': [
                        'scarf', 'shawl',
                        'watch', 'wristwatch',
                        'sunglasses', 'shades',
                        'bag', 'handbag', 'tote', 'shoulder bag',
                        'jewelry', 'necklace', 'bracelet', 'earrings'
                    ]
                }

        
        # Flatten all categories
        self.all_garments = []
        for items in self.garment_categories.values():
            self.all_garments.extend(items)
        
        # Formality levels
        self.formality_types = [
            'formal business attire',
            'business casual',
            'smart casual',
            'casual everyday wear',
            'athletic sportswear',
            'streetwear style'
        ]
        
        # Pre-encode text prompts for efficiency
        self._encode_prompts()
    
    def _encode_prompts(self):
        """Pre-encode all text prompts for faster inference"""
        # Garment prompts with context
        garment_prompts = [f"a photo of a person wearing {g}" 
                          for g in self.all_garments]
        
        with torch.no_grad():
            garment_tokens = clip.tokenize(garment_prompts).to(self.device)
            self.garment_embeddings = self.model.encode_text(garment_tokens)
            self.garment_embeddings /= self.garment_embeddings.norm(dim=-1, keepdim=True)
            
            # Formality embeddings
            formality_tokens = clip.tokenize(self.formality_types).to(self.device)
            self.formality_embeddings = self.model.encode_text(formality_tokens)
            self.formality_embeddings /= self.formality_embeddings.norm(dim=-1, keepdim=True)
    
    @torch.no_grad()
    def detect_garments(self, image: Union[str, Image.Image, np.ndarray], 
                       top_k: int = 3, threshold: float = 0.15) -> List[Dict]:
        """
        Detect top-k garment types in image
        
        Args:
            image: Input image
            top_k: Number of top garments to return
            threshold: Minimum confidence threshold
            
        Returns:
            List of dicts: [{'garment': 'shirt', 'confidence': 0.85, 'category': 'tops'}, ...]
        """
        # Load and preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Encode image
        image_features = self.model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarities with all garments
        similarities = (image_features @ self.garment_embeddings.T).squeeze(0)
        similarities = similarities.cpu().numpy()
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            confidence = float(similarities[idx])
            if confidence >= threshold:
                garment = self.all_garments[idx]
                # Find category
                category = self._find_category(garment)
                results.append({
                    'garment': garment,
                    'confidence': confidence,
                    'category': category
                })
        
        return results
    
    def _find_category(self, garment: str) -> str:
        """Find which category a garment belongs to"""
        for category, items in self.garment_categories.items():
            if garment in items:
                return category
        return 'other'
    
    @torch.no_grad()
    def detect_formality(self, image: Union[str, Image.Image, np.ndarray]) -> Dict[str, float]:
        """
        Detect formality level of outfit
        
        Returns:
            Dict of formality types with confidences
        """
        # Load and preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Encode image
        image_features = self.model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarities
        similarities = (image_features @ self.formality_embeddings.T).squeeze(0)
        
        # Softmax to get probabilities
        probs = torch.softmax(similarities * 100, dim=0).cpu().numpy()
        
        return {style: float(prob) for style, prob in zip(self.formality_types, probs)}
    
    def encode_garments(self, garment_list: List[Dict], dim: int = 128) -> np.ndarray:
        """
        Encode detected garments into fixed-dimensional vector
        
        Args:
            garment_list: Output from detect_garments()
            dim: Output dimension
            
        Returns:
            128-d embedding
        """
        embedding = np.zeros(dim)
        
        if not garment_list:
            return embedding
        
        # Encode top garments with their confidences
        for i, item in enumerate(garment_list[:4]):  # Top 4 garments
            garment_idx = self.all_garments.index(item['garment'])
            category_idx = list(self.garment_categories.keys()).index(item['category'])
            
            # Distribute information across embedding
            base_idx = i * 32
            if base_idx + 32 <= dim:
                embedding[base_idx] = garment_idx / len(self.all_garments)  # Normalized index
                embedding[base_idx + 1] = item['confidence']  # Confidence
                embedding[base_idx + 2] = category_idx / len(self.garment_categories)  # Category
                embedding[base_idx + 3:base_idx + 32] = np.random.randn(29) * 0.05  # Noise
        
        return embedding
    
    def detect_and_encode(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Full pipeline: detect garments and encode to vector
        """
        garments = self.detect_garments(image, top_k=4)
        return self.encode_garments(garments)


# Test it
if __name__ == "__main__":
    detector = GarmentDetector()
    
    # Use config just like other files to avoid path errors
    from utils.config import config
    test_img = config.RAW_DATA_DIR / "download (30).jpg"
    
    # Detect garments
    if test_img.exists():
        garments = detector.detect_garments(test_img, top_k=5)
        print("\nDetected garments:")
        for g in garments:
            print(f"  {g['garment']} ({g['category']}): {g['confidence']:.3f}")
    else:
        print(f"Error: File not found at {test_img}")
        print("Please check the filename in data/raw/")
    
    # Detect formality
    formality = detector.detect_formality(test_img)
    print("\nFormality levels:")
    for style, prob in sorted(formality.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {style}: {prob:.3f}")
    
    # Get embedding
    garment_emb = detector.detect_and_encode(test_img)
    print(f"\nGarment embedding shape: {garment_emb.shape}")