import numpy as np
import faiss
import json
from pathlib import Path
from typing import List, Dict, Tuple
import sys
sys.path.append('..')

from models.clip_encoder import CLIPEncoder
from models.color_extractor import ColorExtractor
from retriever.query_parser import QueryParser
from indexer.fusion import QueryFusion
from utils.config import config
from utils.logger import logger

class FashionSearchEngine:
    """
    Main search engine for fashion retrieval
    Handles query processing and retrieval with fusion strategies
    """
    
    def __init__(self, index_path: str = None, metadata_path: str = None):
        # Load FAISS index
        if index_path is None:
            index_path = config.PROCESSED_DATA_DIR / "faiss_index.bin"
        if metadata_path is None:
            metadata_path = config.PROCESSED_DATA_DIR / "metadata.json"
        
        logger.info("Loading search engine...")
        
        # Load index
        self.index = faiss.read_index(str(index_path))
        logger.info(f"✓ Loaded FAISS index: {self.index.ntotal} vectors")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            self.image_paths = data['image_paths']
            self.metadata_list = data['metadata']
        logger.info(f"✓ Loaded metadata: {len(self.image_paths)} images")
        
        # Initialize models for query encoding
        self.clip_encoder = CLIPEncoder()
        self.color_extractor = ColorExtractor()
        self.query_parser = QueryParser()
        self.query_fusion = QueryFusion()
        
        logger.info("✓ Search engine ready!\n")
    
    def encode_query(self, query: str, weights: Dict = None) -> Tuple[np.ndarray, Dict]:
        """
        Encode text query into retrieval vector using fusion
        
        Uses adaptive multi-modal encoding with QueryFusion
        """
        # Parse query
        parsed = self.query_parser.parse_query(query)
        
        # Build query embeddings for each modality
        query_embeddings = {}
        
        # 1. CLIP embedding (global)
        query_embeddings['clip'] = self.clip_encoder.encode_text(query)
        
        # 2. Color embedding
        query_embeddings['color'] = self._encode_color_query(parsed['colors'])
        
        # 3. Garment embedding
        query_embeddings['garment'] = self._encode_garment_query(parsed['garments'])
        
        # 4. Scene embedding
        query_embeddings['scene'] = self._encode_scene_query(parsed['scenes'])
        
        # Fuse using QueryFusion
        if weights is None:
            # Use adaptive query-specific fusion
            query_vector = self.query_fusion.fuse_query(
                query_embeddings,
                parsed,
                use_adaptive=True
            )
            used_weights = self.query_fusion._compute_query_weights(parsed)
        else:
            # Use provided weights
            from indexer.fusion import EmbeddingFusion
            fusion = EmbeddingFusion()
            query_vector = fusion.fuse_embeddings(query_embeddings, weights)
            used_weights = weights
        
        # Add fusion explanation to parsed query
        parsed['fusion_weights'] = used_weights
        parsed['fusion_explanation'] = self.query_fusion.explain_weights(parsed)
        
        return query_vector, parsed
    
    def _encode_color_query(self, colors: List[str]) -> np.ndarray:
        """
        Encode color terms using CLIP text embeddings (semantic)
        """
        color_emb = np.zeros(config.EMBEDDING_DIM['color'])
        
        if not colors:
            return color_emb
        
        # Use CLIP for semantic color encoding (max 2 colors for 64-dim space)
        for i, color in enumerate(colors[:2]):
            # Create rich color description
            color_query = f"{color} colored clothing"
            clip_color = self.clip_encoder.encode_text(color_query)
            
            # Take first 32 dims of CLIP embedding for this color
            start_idx = i * 32
            if start_idx + 32 <= len(color_emb):
                color_emb[start_idx:start_idx + 32] = clip_color[:32]
        
        return color_emb
    
    def _encode_garment_query(self, garments: List[str]) -> np.ndarray:
        """
        Encode garment terms with multiple prompt strategies
        """
        garment_emb = np.zeros(config.EMBEDDING_DIM['garment'])
        
        if not garments:
            return garment_emb
        
        for i, garment in enumerate(garments[:4]):
            # Multiple prompts for robustness
            prompts = [
                f"a person wearing a {garment}",
                f"someone in a {garment}",
                f"a {garment}"
            ]
            
            # Average embeddings from multiple prompts
            clip_garments = [self.clip_encoder.encode_text(p) for p in prompts]
            avg_garment = np.mean(clip_garments, axis=0)
            
            base_idx = i * 32
            if base_idx + 32 <= len(garment_emb):
                garment_emb[base_idx:base_idx + 32] = avg_garment[:32]
        
        return garment_emb
    
    def _encode_scene_query(self, scenes: List[str]) -> np.ndarray:
        """
        Encode scene/location terms with rich descriptions
        """
        scene_emb = np.zeros(config.EMBEDDING_DIM['scene'])
        
        if not scenes:
            return scene_emb
        
        for i, scene in enumerate(scenes[:4]):
            # Richer scene descriptions
            prompts = [
                f"a photo taken in {scene}",
                f"{scene} setting",
                f"{scene} environment"
            ]
            
            clip_scenes = [self.clip_encoder.encode_text(p) for p in prompts]
            avg_scene = np.mean(clip_scenes, axis=0)
            
            base_idx = i * 32
            if base_idx + 32 <= len(scene_emb):
                scene_emb[base_idx:base_idx + 32] = avg_scene[:32]
        
        return scene_emb
    
    def search(self, query: str, top_k: int = 10, 
               weights: Dict = None, verbose: bool = False) -> Tuple[List[Dict], Dict]:
        """
        Search for images matching query
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            weights: Optional custom fusion weights
            verbose: Print fusion details
            
        Returns:
            (results, parsed_query_with_fusion_info)
        """
        # Encode query with fusion
        query_vector, parsed_query = self.encode_query(query, weights)
        
        if verbose:
            logger.info(f"\n🔍 Query Processing:")
            logger.info(f"  Raw: {query}")
            logger.info(f"  Parsed colors: {parsed_query['colors']}")
            logger.info(f"  Parsed garments: {parsed_query['garments']}")
            logger.info(f"  Parsed scenes: {parsed_query['scenes']}")
            logger.info(f"\n{parsed_query['fusion_explanation']}")
        
        # Search FAISS index
        query_vector = query_vector.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        
        # Prepare results
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.image_paths):  # Valid index
                result = {
                    'rank': rank + 1,
                    'image_path': self.image_paths[idx],
                    'score': float(score),
                    'metadata': self.metadata_list[idx]
                }
                results.append(result)
        
        return results, parsed_query
    
    def search_with_rerank(self, query: str, top_k: int = 10) -> Tuple[List[Dict], Dict]:
        """
        Two-stage retrieval: CLIP-based retrieval + attribute re-ranking
        BONUS: Use this if standard search still gives poor results
        """
        # Stage 1: Get top-50 with current method
        results, parsed = self.search(query, top_k=50)
        
        # Stage 2: Re-rank by explicit attribute matching
        for result in results:
            metadata = result['metadata']
            
            # Color match bonus
            color_bonus = 0
            for qcolor in parsed['colors']:
                base_color = qcolor.split()[-1]  # Remove modifiers
                if base_color in metadata['colors']:
                    color_bonus += 0.1
            
            # Garment match bonus
            garment_bonus = 0
            image_garments = [g['garment'].lower() for g in metadata['garments']]
            for qgarment in parsed['garments']:
                for igarment in image_garments:
                    if qgarment.lower() in igarment or igarment in qgarment.lower():
                        garment_bonus += 0.15
                        break
            
            # Scene match bonus
            scene_bonus = 0
            image_scenes = [s['scene'].lower() for s in metadata['scenes']]
            for qscene in parsed['scenes']:
                for iscene in image_scenes:
                    if qscene.lower() in iscene or iscene in qscene.lower():
                        scene_bonus += 0.1
                        break
            
            # Adjust score with bonuses
            result['original_score'] = result['score']
            result['score'] = result['score'] + color_bonus + garment_bonus + scene_bonus
        
        # Re-sort by new scores
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Re-assign ranks
        for i, r in enumerate(results[:top_k]):
            r['rank'] = i + 1
        
        return results[:top_k], parsed
    
    def search_with_explanation(self, query: str, top_k: int = 10, use_rerank: bool = False) -> Dict:
        """
        Search with detailed explanation of query processing and fusion
        
        Args:
            use_rerank: If True, uses two-stage retrieval with re-ranking
        """
        if use_rerank:
            results, parsed = self.search_with_rerank(query, top_k)
        else:
            results, parsed = self.search(query, top_k, verbose=True)
        
        explanation = {
            'query': query,
            'parsed_query': parsed,
            'fusion_weights': parsed['fusion_weights'],
            'fusion_explanation': parsed['fusion_explanation'],
            'num_results': len(results),
            'results': results,
            'used_reranking': use_rerank
        }
        
        return explanation
    
    def compare_fusion_strategies(self, query: str, top_k: int = 5) -> Dict:
        """
        Compare different fusion strategies on same query
        """
        # 1. Vanilla CLIP only
        vanilla_weights = {'clip': 1.0, 'color': 0.0, 'garment': 0.0, 'scene': 0.0}
        vanilla_results, _ = self.search(query, top_k, weights=vanilla_weights)
        
        # 2. Equal weighting
        equal_weights = {'clip': 0.25, 'color': 0.25, 'garment': 0.25, 'scene': 0.25}
        equal_results, _ = self.search(query, top_k, weights=equal_weights)
        
        # 3. Adaptive (our approach)
        adaptive_results, parsed = self.search(query, top_k)
        
        # 4. Adaptive with re-ranking
        rerank_results, _ = self.search_with_rerank(query, top_k)
        
        return {
            'query': query,
            'vanilla_clip': {
                'results': vanilla_results,
                'weights': vanilla_weights
            },
            'equal_weighting': {
                'results': equal_results,
                'weights': equal_weights
            },
            'adaptive_fusion': {
                'results': adaptive_results,
                'weights': parsed['fusion_weights'],
                'explanation': parsed['fusion_explanation']
            },
            'adaptive_with_rerank': {
                'results': rerank_results,
                'weights': parsed['fusion_weights']
            },
            'parsed_query': parsed
        }


# Test it
if __name__ == "__main__":
    engine = FashionSearchEngine()
    
    test_queries = [
        "A person in a bright yellow raincoat",
        "Professional business attire inside a modern office",
        "Someone wearing a blue shirt sitting on a park bench",
        "Casual weekend outfit for a city walk",
        "A red tie and a white shirt in a formal setting"
    ]
    
    logger.info("\n" + "="*60)
    logger.info("Testing IMPROVED Search Engine")
    logger.info("="*60)
    
    for query in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Query: {query}")
        logger.info('='*60)
        
        # Test with re-ranking for compositional queries
        use_rerank = any(word in query.lower() for word in ['and', 'with', 'in a'])
        
        explanation = engine.search_with_explanation(query, top_k=5, use_rerank=use_rerank)
        
        logger.info(f"\nTop 5 Results:")
        for r in explanation['results']:
            logger.info(f"\n{r['rank']}. {Path(r['image_path']).name}")
            logger.info(f"   Score: {r['score']:.4f}")
            logger.info(f"   Colors: {', '.join(r['metadata']['colors'][:3])}")
            garment_names = [g['garment'] for g in r['metadata']['garments'][:3]]
            logger.info(f"   Garments: {', '.join(garment_names)}")
