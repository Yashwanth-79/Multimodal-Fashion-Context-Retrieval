import numpy as np
from pathlib import Path
from typing import Union, Dict
from PIL import Image
import sys
sys.path.append('..')

from models.clip_encoder import CLIPEncoder
from models.color_extractor import ColorExtractor
from models.garment_detector import GarmentDetector
from models.scene_classifier import SceneClassifier
from indexer.fusion import EmbeddingFusion
from utils.config import config

class MultimodalFeatureExtractor:
    """
    Unified feature extraction pipeline
    Combines CLIP, color, garment, and scene features using fusion strategies
    """
    
    def __init__(self, device: str = None, fusion_type: str = "weighted_concat"):
        print("Initializing MultimodalFeatureExtractor...")
        
        # Initialize all models
        self.clip_encoder = CLIPEncoder(device=device)
        self.color_extractor = ColorExtractor(device=device)
        self.garment_detector = GarmentDetector(clip_encoder=self.clip_encoder, device=device)
        self.scene_classifier = SceneClassifier(clip_encoder=self.clip_encoder, device=device)
        
        # Initialize fusion module
        self.fusion = EmbeddingFusion(fusion_type=fusion_type)
        
        self.device = device
        print("✓ All models loaded successfully!")
        print(f"✓ Using fusion strategy: {fusion_type}")
    
    def extract_features(self, image_path: Union[str, Path]) -> Dict:
        """
        Extract all features from a single image
        
        Returns:
            Dict containing:
                - embeddings: Dict of individual modality embeddings
                - combined_embedding: Fused 832-d vector
                - metadata: extracted attributes
        """
        image_path = str(image_path)
        
        # Extract all features
        clip_emb = self.clip_encoder.encode_image(image_path)
        color_emb = self.color_extractor.extract_and_encode(image_path)
        garment_emb = self.garment_detector.detect_and_encode(image_path)
        scene_emb = self.scene_classifier.classify_and_encode(image_path)
        
        # Store individual embeddings
        embeddings = {
            'clip': clip_emb,
            'color': color_emb,
            'garment': garment_emb,
            'scene': scene_emb
        }
        
        # Extract metadata for interpretability
        colors = self.color_extractor.get_color_names(image_path)
        garments = self.garment_detector.detect_garments(image_path, top_k=3)
        scenes = self.scene_classifier.classify_scene(image_path, top_k=2)
        
        metadata = {
            'colors': colors,
            'garments': garments,
            'scenes': scenes,
            'image_path': image_path
        }
        
        return {
            'embeddings': embeddings,
            'metadata': metadata
        }
    
    def combine_embeddings(self, embeddings: Dict[str, np.ndarray], 
                          weights: Dict = None,
                          metadata: Dict = None) -> np.ndarray:
        """
        Combine all embeddings with fusion strategy
        
        Args:
            embeddings: Dict of individual modality embeddings
            weights: Optional custom weights
            metadata: Optional metadata for adaptive weighting
        """
        # Use adaptive weights if metadata provided
        if weights is None and metadata is not None:
            weights = self.fusion.compute_adaptive_weights(metadata)
        
        # Fuse embeddings
        combined = self.fusion.fuse_embeddings(embeddings, weights)
        
        return combined
    
    def extract_and_combine(self, image_path: Union[str, Path], 
                           weights: Dict = None,
                           use_adaptive: bool = True) -> tuple:
        """
        Full pipeline: extract and combine features
        
        Args:
            image_path: Path to image
            weights: Optional custom weights
            use_adaptive: Whether to use adaptive weighting based on image content
            
        Returns:
            (combined_embedding, metadata, individual_embeddings)
        """
        # Extract all features
        features = self.extract_features(image_path)
        
        # Combine with optional adaptive weighting
        if use_adaptive and weights is None:
            combined_emb = self.combine_embeddings(
                features['embeddings'],
                metadata=features['metadata']
            )
        else:
            combined_emb = self.combine_embeddings(
                features['embeddings'],
                weights=weights
            )
        
        return combined_emb, features['metadata'], features['embeddings']


# Test it
if __name__ == "__main__":
    import time
    
    extractor = MultimodalFeatureExtractor()
    
    test_img = "data/raw/test.jpg"
    
    print("\n" + "="*60)
    print("Testing MultimodalFeatureExtractor with Fusion")
    print("="*60)
    
    start = time.time()
    combined_emb, metadata, individual_embs = extractor.extract_and_combine(
        test_img, 
        use_adaptive=True
    )
    elapsed = time.time() - start
    
    print(f"\n✓ Processing time: {elapsed:.2f}s")
    print(f"✓ Combined embedding shape: {combined_emb.shape}")
    
    print(f"\n📊 Individual Embedding Shapes:")
    for modality, emb in individual_embs.items():
        print(f"  {modality}: {emb.shape}")
    
    print(f"\n📊 Extracted Metadata:")
    print(f"  Colors: {metadata['colors']}")
    print(f"  Garments:")
    for g in metadata['garments']:
        print(f"    - {g['garment']} ({g['confidence']:.2f})")
    print(f"  Scenes:")
    for s in metadata['scenes']:
        print(f"    - {s['scene']} ({s['confidence']:.2f})")
    
    # Test fusion explanation
    print(f"\n💡 Fusion Explanation:")
    from indexer.fusion import QueryFusion
    qf = QueryFusion()
    mock_parsed = {
        'colors': metadata['colors'][:2],
        'garments': [g['garment'] for g in metadata['garments'][:1]],
        'scenes': [s['scene'] for s in metadata['scenes'][:1]],
        'is_compositional': True
    }
    print(qf.explain_weights(mock_parsed))