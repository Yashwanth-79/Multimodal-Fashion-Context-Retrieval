import cv2
import numpy as np
from PIL import Image
import torch
import clip
from typing import List, Tuple, Union
from pathlib import Path

class ColorExtractor:
    """
    CLIP-based color extraction - semantically understands colors in context
   
    """
    
    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load CLIP
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # Color vocabulary
        self.color_vocab = [
            'red', 'blue', 'green', 'yellow', 'orange', 'purple',
            'pink', 'brown', 'black', 'white', 'gray', 
            'navy', 'maroon', 'beige', 'teal', 'coral', 'gold'
        ]
        
        # Pre-encode color prompts for efficiency
        self._precompute_color_embeddings()
    
    def _precompute_color_embeddings(self):
        """Pre-encode all color queries for fast inference"""
        # Create rich color prompts
        color_prompts = []
        for color in self.color_vocab:
            # Multiple prompt variations for robustness
            prompts = [
                f"clothing in {color} color",
                f"{color} garment",
                f"person wearing {color}",
                f"{color} shirt",
                f"{color} dress"
            ]
            color_prompts.extend(prompts)
        
        # Encode all prompts
        with torch.no_grad():
            text_tokens = clip.tokenize(color_prompts).to(self.device)
            self.color_embeddings = self.model.encode_text(text_tokens)
            self.color_embeddings /= self.color_embeddings.norm(dim=-1, keepdim=True)
        
        # Store mapping of embeddings to color names
        self.prompt_to_color = []
        for color in self.color_vocab:
            self.prompt_to_color.extend([color] * 5)  # 5 prompts per color
    
    @torch.no_grad()
    def get_color_names(self, image: Union[str, Image.Image, np.ndarray], 
                       top_k: int = 5, threshold: float = 0.20) -> List[str]:
        """
        Extract colors using CLIP semantic understanding
        
        Args:
            image: Input image
            top_k: Number of colors to return
            threshold: Minimum confidence threshold
            
        Returns:
            List of color names detected in the image
        """
        # Load and preprocess image
        if isinstance(image, (str, Path)):
            image = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Encode image
        image_features = self.model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarities with all color prompts
        similarities = (image_features @ self.color_embeddings.T).squeeze(0)
        similarities = similarities.cpu().numpy()
        
        # Aggregate scores by color (average across 5 prompts)
        color_scores = {}
        for i, color in enumerate(self.prompt_to_color):
            if color not in color_scores:
                color_scores[color] = []
            color_scores[color].append(similarities[i])
        
        # Average scores for each color
        color_avg_scores = {
            color: np.mean(scores) 
            for color, scores in color_scores.items()
        }
        
        # Sort by score
        sorted_colors = sorted(
            color_avg_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Filter by threshold and return top-k
        detected_colors = []
        for color, score in sorted_colors:
            if score >= threshold and len(detected_colors) < top_k:
                detected_colors.append(color)
        
        # If no colors pass threshold, return top 3 anyway
        if not detected_colors:
            detected_colors = [color for color, _ in sorted_colors[:3]]
        
        return detected_colors
    
    def get_color_scores(self, image: Union[str, Image.Image, np.ndarray]) -> dict:
        """
        Get confidence scores for all colors
        Useful for debugging
        """
        if isinstance(image, (str, Path)):
            image = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            similarities = (image_features @ self.color_embeddings.T).squeeze(0)
            similarities = similarities.cpu().numpy()
        
        # Aggregate by color
        color_scores = {}
        for i, color in enumerate(self.prompt_to_color):
            if color not in color_scores:
                color_scores[color] = []
            color_scores[color].append(float(similarities[i]))
        
        # Average
        return {
            color: np.mean(scores) 
            for color, scores in color_scores.items()
        }
    
    def encode_colors(self, color_names: List[str], dim: int = 64) -> np.ndarray:
        """
        Encode color names to fixed-dimensional vector
        
        """
        embedding = np.zeros(dim)
        
        if not color_names:
            return embedding
        
        # Use CLIP embeddings for semantic encoding
        for i, color in enumerate(color_names[:2]):  # Max 2 colors for 64-dim
            # Create color query
            color_text = f"{color} colored clothing"
            
            with torch.no_grad():
                text_token = clip.tokenize([color_text]).to(self.device)
                text_embedding = self.model.encode_text(text_token)
                text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
                text_embedding = text_embedding.cpu().numpy().squeeze()
            
            # Take first 32 dims
            start_idx = i * 32
            if start_idx + 32 <= dim:
                embedding[start_idx:start_idx + 32] = text_embedding[:32]
        
        return embedding
    
    def extract_and_encode(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """Full pipeline: extract colors and encode to vector"""
        color_names = self.get_color_names(image, top_k=5)
        return self.encode_colors(color_names)


# Test it
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from pathlib import Path
    
    print("\n" + "="*60)
    print("Testing CLIP-based Color Extractor")
    print("="*60)
    
    extractor = ColorExtractor()
    
    # Use config for absolute path to avoid path errors
    from utils.config import config
    test_img = config.RAW_DATA_DIR / "download (14).jpg"
    
    if test_img.exists():
        print(f"\nTesting on: {test_img}")
        
        # Get color scores (for debugging)
        scores = extractor.get_color_scores(test_img)
        print("\n📊 Color Confidence Scores:")
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for color, score in sorted_scores[:10]:
            print(f"  {color}: {score:.4f}")
        
        # Get detected colors
        colors = extractor.get_color_names(test_img, top_k=5)
        print(f"\n🎨 Detected Colors: {colors}")
        
        print("\n✅ Should now detect RED and WHITE for tie image!")
    else:
        print(f"⚠️  Test image not found: {test_img}")
        print("Testing with random image...")
        
        # Test with any image
        import os
        data_dir = Path("data/raw")
        if data_dir.exists():
            test_imgs = list(data_dir.glob("*.jpg"))[:3]
            for img in test_imgs:
                colors = extractor.get_color_names(str(img))
                print(f"\n{img.name}: {colors}")
