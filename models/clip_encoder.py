import torch
import clip
from PIL import Image
import numpy as np
from typing import Union, List

class CLIPEncoder:
    """CLIP model wrapper for image and text encoding"""
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
    @torch.no_grad()
    def encode_image(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Encode single image to embedding
        
        Args:
            image: Path to image, PIL Image, or numpy array
            
        Returns:
            512-dimensional embedding
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
            
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image_input)
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().flatten()
    
    @torch.no_grad()
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text to embedding
        
        Args:
            text: Single string or list of strings
            
        Returns:
            512-dimensional embedding (or batch)
        """
        if isinstance(text, str):
            text = [text]
            
        text_tokens = clip.tokenize(text).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        
        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        result = text_features.cpu().numpy()
        return result[0] if len(text) == 1 else result
    
    @torch.no_grad()
    def compute_similarity(self, image_emb: np.ndarray, text_emb: np.ndarray) -> float:
        """Compute cosine similarity between image and text embeddings"""
        return float(np.dot(image_emb, text_emb))
    
    def zero_shot_classify(self, image: Union[str, Image.Image], 
                          categories: List[str]) -> dict:
        """
        Zero-shot classification of image into categories
        
        Returns:
            dict: {category: probability}
        """
        image_emb = self.encode_image(image)
        text_embs = self.encode_text(categories)
        
        # Compute similarities
        similarities = image_emb @ text_embs.T
        probs = np.exp(similarities) / np.exp(similarities).sum()
        
        return {cat: float(prob) for cat, prob in zip(categories, probs)}


# Test it
if __name__ == "__main__":
    encoder = CLIPEncoder()
    
    # Test image encoding
    test_img = "data/raw/test.jpg"  # Replace with your image
    img_emb = encoder.encode_image(test_img)
    print(f"Image embedding shape: {img_emb.shape}")
    
    # Test text encoding
    text_emb = encoder.encode_text("a person wearing a red dress")
    print(f"Text embedding shape: {text_emb.shape}")
    
    # Test similarity
    sim = encoder.compute_similarity(img_emb, text_emb)
    print(f"Similarity: {sim:.4f}")