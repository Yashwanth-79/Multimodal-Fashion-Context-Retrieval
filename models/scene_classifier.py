import torch
import numpy as np
from typing import Union, List, Dict
from PIL import Image
import clip

class SceneClassifier:
    """
    Classify scene/environment context using CLIP
    Handles indoor/outdoor, specific locations
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
        
        # Scene categories (hierarchical)
        self.scene_categories = {

                'office': [
                    'office', 'office interior', 'modern office', 'corporate office',
                    'workspace', 'workplace', 'open office',
                    'meeting room', 'conference room',
                    'office building', 'business center'
                ],

                'urban_street': [
                    'city street', 'urban street', 'street',
                    'downtown', 'city center',
                    'sidewalk', 'crosswalk',
                    'plaza', 'square',
                    'urban area', 'city walk'
                ],

                'park': [
                    'park', 'city park',
                    'garden', 'public garden',
                    'green space', 'outdoor park',
                    'park bench', 'lawn', 'pathway'
                ],

                'home': [
                    'home', 'home interior',
                    'house', 'apartment', 'flat',
                    'living room', 'bedroom',
                    'kitchen', 'balcony',
                    'indoor home', 'residential interior'
                ]
        }
        
        # Flatten for easy access
        self.all_scenes = []
        self.scene_to_category = {}
        
        for category, scenes in self.scene_categories.items():
            for scene in scenes:
                self.all_scenes.append(scene)
                self.scene_to_category[scene] = {
                    'environment': category,
                    'location': category
                }
        
        # Weather/lighting conditions
        self.conditions = [
            'bright sunny day',
            'overcast cloudy',
            'rainy weather',
            'snowy weather',
            'foggy misty',
            'nighttime evening',
            'golden hour sunset',
            'indoor artificial lighting'
        ]
        
        # Pre-encode prompts
        self._encode_prompts()
    
    def _encode_prompts(self):
        """Pre-encode all scene prompts"""
        scene_prompts = [f"a photo taken in {s}" for s in self.all_scenes]
        
        with torch.no_grad():
            scene_tokens = clip.tokenize(scene_prompts).to(self.device)
            self.scene_embeddings = self.model.encode_text(scene_tokens)
            self.scene_embeddings /= self.scene_embeddings.norm(dim=-1, keepdim=True)
            
            # Condition embeddings
            condition_tokens = clip.tokenize(self.conditions).to(self.device)
            self.condition_embeddings = self.model.encode_text(condition_tokens)
            self.condition_embeddings /= self.condition_embeddings.norm(dim=-1, keepdim=True)
    
    @torch.no_grad()
    def classify_scene(self, image: Union[str, Image.Image, np.ndarray],
                      top_k: int = 3) -> List[Dict]:
        """
        Classify scene/location of image
        
        Returns:
            List of dicts with scene info and confidence
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
        similarities = (image_features @ self.scene_embeddings.T).squeeze(0)
        similarities = similarities.cpu().numpy()
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            scene = self.all_scenes[idx]
            confidence = float(similarities[idx])
            category_info = self.scene_to_category[scene]
            
            results.append({
                'scene': scene,
                'confidence': confidence,
                'environment': category_info['environment'],
                'location': category_info['location']
            })
        
        return results
    
    @torch.no_grad()
    def detect_conditions(self, image: Union[str, Image.Image, np.ndarray]) -> Dict[str, float]:
        """
        Detect weather/lighting conditions
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
        similarities = (image_features @ self.condition_embeddings.T).squeeze(0)
        probs = torch.softmax(similarities * 100, dim=0).cpu().numpy()
        
        return {cond: float(prob) for cond, prob in zip(self.conditions, probs)}
    
    @torch.no_grad()
    def detect_indoor_outdoor(self, image: Union[str, Image.Image, np.ndarray]) -> Dict[str, float]:
        """
        Simple indoor vs outdoor classification
        """
        prompts = ["indoor photo", "outdoor photo"]
        
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize(prompts).to(self.device)
        
        image_features = self.model.encode_image(image_input)
        text_features = self.model.encode_text(text_tokens)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarities = (image_features @ text_features.T).squeeze(0)
        probs = torch.softmax(similarities * 100, dim=0).cpu().numpy()
        
        return {'indoor': float(probs[0]), 'outdoor': float(probs[1])}
    
    def encode_scene(self, scene_list: List[Dict], dim: int = 128) -> np.ndarray:
        """
        Encode detected scenes into fixed-dimensional vector
        """
        embedding = np.zeros(dim)
        
        if not scene_list:
            return embedding
        
        for i, item in enumerate(scene_list[:4]):  # Top 4 scenes
            scene_idx = self.all_scenes.index(item['scene'])
            env_idx = 0 if item['environment'] == 'indoor' else 1
            
            base_idx = i * 32
            if base_idx + 32 <= dim:
                embedding[base_idx] = scene_idx / len(self.all_scenes)
                embedding[base_idx + 1] = item['confidence']
                embedding[base_idx + 2] = env_idx
                embedding[base_idx + 3:base_idx + 32] = np.random.randn(29) * 0.05
        
        return embedding
    
    def classify_and_encode(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Full pipeline: classify scene and encode to vector
        """
        scenes = self.classify_scene(image, top_k=3)
        return self.encode_scene(scenes)


# Test it
if __name__ == "__main__":
    classifier = SceneClassifier()
    
    test_img = "data/raw/test.jpg"
    
    # Classify scene
    scenes = classifier.classify_scene(test_img, top_k=5)
    print("\nDetected scenes:")
    for s in scenes:
        print(f"  {s['scene']} ({s['environment']}/{s['location']}): {s['confidence']:.3f}")
    
    # Indoor/outdoor
    in_out = classifier.detect_indoor_outdoor(test_img)
    print(f"\nIndoor/Outdoor: {in_out}")
    
    # Conditions
    conditions = classifier.detect_conditions(test_img)
    print("\nTop conditions:")
    for cond, prob in sorted(conditions.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {cond}: {prob:.3f}")
    
    # Get embedding
    scene_emb = classifier.classify_and_encode(test_img)
    print(f"\nScene embedding shape: {scene_emb.shape}")