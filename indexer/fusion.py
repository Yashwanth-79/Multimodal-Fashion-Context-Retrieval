"""
Fusion strategies for combining multimodal embeddings
Implements adaptive weighting and multiple fusion techniques
"""

import numpy as np
from typing import Dict, List, Optional
import sys
sys.path.append('..')
from utils.config import config


class EmbeddingFusion:
    """
    Handles fusion of multiple embedding modalities
    Supports multiple fusion strategies
    """
    
    def __init__(self, fusion_type: str = "weighted_concat"):
        """
        Args:
            fusion_type: Type of fusion
                - "weighted_concat": Weighted concatenation (default)
                - "late_fusion": Separate search then merge
                - "learned_fusion": Trainable fusion (future work)
        """
        self.fusion_type = fusion_type
        self.default_weights = config.DEFAULT_WEIGHTS.copy()
        
    def fuse_embeddings(self, 
                       embeddings: Dict[str, np.ndarray],
                       weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Fuse multiple embeddings into single vector
        
        Args:
            embeddings: Dict mapping modality name to embedding
                e.g., {'clip': array, 'color': array, 'garment': array, 'scene': array}
            weights: Optional custom weights for each modality
            
        Returns:
            Fused embedding vector
        """
        if weights is None:
            weights = self.default_weights
        
        if self.fusion_type == "weighted_concat":
            return self._weighted_concatenation(embeddings, weights)
        elif self.fusion_type == "late_fusion":
            return self._late_fusion(embeddings, weights)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
    
    def _weighted_concatenation(self, 
                               embeddings: Dict[str, np.ndarray],
                               weights: Dict[str, float]) -> np.ndarray:
        """
        Weighted concatenation fusion
        
        Each embedding is scaled by its weight then concatenated
        """
        fused_parts = []
        
        # Ensure order: clip, color, garment, scene
        modality_order = ['clip', 'color', 'garment', 'scene']
        
        for modality in modality_order:
            if modality in embeddings:
                weighted_emb = embeddings[modality] * weights.get(modality, 0.25)
                fused_parts.append(weighted_emb)
        
        # Concatenate
        fused = np.concatenate(fused_parts)
        
        # L2 normalize
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm
        
        return fused
    
    def _late_fusion(self,
                    embeddings: Dict[str, np.ndarray],
                    weights: Dict[str, float]) -> np.ndarray:
        """
        Late fusion: Keep embeddings separate for now
        (Used in retrieval stage to search separately then merge scores)
        
        For indexing, we still concatenate but mark boundaries
        """
        # For late fusion, we still need to store all embeddings
        # but the retrieval will handle them differently
        return self._weighted_concatenation(embeddings, weights)
    
    def compute_adaptive_weights(self,
                                metadata: Dict,
                                query_type: str = "general") -> Dict[str, float]:
        """
        Compute adaptive weights based on image metadata
        
        Args:
            metadata: Image metadata with detected attributes
            query_type: Type of query ("color_focused", "garment_focused", etc.)
            
        Returns:
            Adjusted weights
        """
        weights = self.default_weights.copy()
        
        # Adjust based on image characteristics
        num_colors = len(metadata.get('colors', []))
        num_garments = len(metadata.get('garments', []))
        num_scenes = len(metadata.get('scenes', []))
        
        # If image has many detected attributes in a modality, 
        # that modality might be more reliable
        if num_colors >= 3:
            weights['color'] += 0.05
        if num_garments >= 2:
            weights['garment'] += 0.05
        if num_scenes >= 2:
            weights['scene'] += 0.05
        
        # Adjust based on query type
        if query_type == "color_focused":
            weights['color'] = 0.4
            weights['clip'] = 0.3
        elif query_type == "garment_focused":
            weights['garment'] = 0.4
            weights['clip'] = 0.3
        elif query_type == "scene_focused":
            weights['scene'] = 0.4
            weights['clip'] = 0.3
        
        # Normalize to sum to 1.0
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def get_embedding_dimensions(self) -> Dict[str, int]:
        """Return dimensions for each modality"""
        return config.EMBEDDING_DIM.copy()
    
    def split_fused_embedding(self, fused: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split a fused embedding back into components
        
        Useful for debugging or re-weighting
        """
        dims = self.get_embedding_dimensions()
        
        components = {}
        start_idx = 0
        
        for modality in ['clip', 'color', 'garment', 'scene']:
            dim = dims[modality]
            components[modality] = fused[start_idx:start_idx + dim]
            start_idx += dim
        
        return components


class QueryFusion:
    """
    Specialized fusion for query embeddings
    Handles compositional queries differently
    """
    
    def __init__(self):
        self.base_fusion = EmbeddingFusion()
    
    def fuse_query(self,
                   query_embeddings: Dict[str, np.ndarray],
                   parsed_query: Dict,
                   use_adaptive: bool = True) -> np.ndarray:
        """
        Fuse query embeddings with query-specific logic
        
        Args:
            query_embeddings: Dict of modality embeddings
            parsed_query: Parsed query structure from QueryParser
            use_adaptive: Whether to use adaptive weighting
        """
        if use_adaptive:
            weights = self._compute_query_weights(parsed_query)
        else:
            weights = config.DEFAULT_WEIGHTS.copy()
        
        return self.base_fusion.fuse_embeddings(query_embeddings, weights)
    
    def _compute_query_weights(self, parsed_query: Dict) -> Dict[str, float]:
        """
        Compute weights based on query characteristics
        Using CLIP stronger for semantic understanding
        """
        weights = config.DEFAULT_WEIGHTS.copy()
        
        has_colors = len(parsed_query.get('colors', [])) > 0
        has_garments = len(parsed_query.get('garments', [])) > 0
        has_scenes = len(parsed_query.get('scenes', [])) > 0
        is_compositional = parsed_query.get('is_compositional', False)
        
        # Compositional queries: Balance CLIP with attributes
        if is_compositional:
            weights['clip'] = 0.40  # INCREASED from 0.25 - keep semantic understanding
            
            active_modalities = sum([has_colors, has_garments, has_scenes])
            if active_modalities > 0:
                remaining_weight = 0.60  # DECREASED from 0.75
                weight_per_modality = remaining_weight / active_modalities
                
                weights['color'] = weight_per_modality if has_colors else 0.0
                weights['garment'] = weight_per_modality if has_garments else 0.0
                weights['scene'] = weight_per_modality if has_scenes else 0.0
        
        # Simple queries: CLIP dominates with attribute support
        else:
            if has_colors and not has_garments and not has_scenes:
                weights = {'clip': 0.5, 'color': 0.5, 'garment': 0.0, 'scene': 0.0}
            elif has_garments and not has_colors and not has_scenes:
                weights = {'clip': 0.5, 'color': 0.0, 'garment': 0.5, 'scene': 0.0}
            elif has_scenes and not has_colors and not has_garments:
                weights = {'clip': 0.5, 'color': 0.0, 'garment': 0.0, 'scene': 0.5}
            # Otherwise keep defaults
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def explain_weights(self, parsed_query: Dict) -> str:
        """
        Generate human-readable explanation of weight decisions
        """
        weights = self._compute_query_weights(parsed_query)
        
        explanation = "Weight allocation:\n"
        for modality, weight in weights.items():
            if weight > 0:
                explanation += f"  {modality}: {weight:.2%} "
                if modality == 'clip':
                    explanation += "(global semantic understanding)\n"
                elif modality == 'color':
                    explanation += f"(colors: {parsed_query.get('colors', [])})\n"
                elif modality == 'garment':
                    explanation += f"(garments: {parsed_query.get('garments', [])})\n"
                elif modality == 'scene':
                    explanation += f"(scenes: {parsed_query.get('scenes', [])})\n"
        
        return explanation


# Test the fusion module
if __name__ == "__main__":
    print("="*60)
    print("Testing IMPROVED Fusion Module")
    print("="*60)
    
    # Test embedding fusion
    fusion = EmbeddingFusion()
    
    # Mock embeddings
    mock_embeddings = {
        'clip': np.random.randn(512),
        'color': np.random.randn(64),
        'garment': np.random.randn(128),
        'scene': np.random.randn(128)
    }
    
    # Test default fusion
    fused = fusion.fuse_embeddings(mock_embeddings)
    print(f"\n✓ Fused embedding shape: {fused.shape}")
    print(f"✓ Expected shape: {config.TOTAL_DIM}")
    
    # Test query fusion
    query_fusion = QueryFusion()
    
    # Test compositional query
    mock_parsed = {
        'colors': ['red', 'white'],
        'garments': ['shirt', 'tie'],
        'scenes': [],
        'is_compositional': True
    }
    
    query_weights = query_fusion._compute_query_weights(mock_parsed)
    print(f"\n✓ Compositional query weights (IMPROVED):")
    print(f"   {query_weights}")
    print(f"   Note: CLIP at {query_weights['clip']:.1%} (was 25%, now 40%)")
    
    # Test explanation
    explanation = query_fusion.explain_weights(mock_parsed)
    print(f"\n{explanation}")