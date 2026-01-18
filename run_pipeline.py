#!/usr/bin/env python3
"""
Complete pipeline to build and test fashion retrieval system
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from indexer.build_index import IndexBuilder
from retriever.search_engine import FashionSearchEngine
from utils.logger import logger

def build_index():
    """Build the index from images"""
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Building Index")
    logger.info("="*60 + "\n")
    
    builder = IndexBuilder(data_dir="data/raw")
    builder.build_complete_index(
        index_type="IVF",  
        output_dir="data/processed"
    )

def test_retrieval():
    """Test the retrieval system"""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Testing Retrieval")
    logger.info("="*60 + "\n")
    
    engine = FashionSearchEngine()
    
    # Evaluation queries from assignment
    test_queries = [
         "A person in a bright yellow raincoat",
        # "Professional business attire inside a modern office",
        # "Someone wearing a blue shirt sitting on a park bench",
        # "Casual weekend outfit for a city walk",
    #    "A red tie and a white shirt in a formal setting"
    ]
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Test Query {i}: {query}")
        logger.info('='*60)
        
        # Auto-enable re-ranking for complex queries
        use_rerank = any(word in query.lower() for word in ['and', 'with', 'in a'])
        if use_rerank:
            logger.info("ℹ️  Complex query detected: Enabling Attribute Re-ranking")

        # Get results with explanation
        explanation = engine.search_with_explanation(query, top_k=10, use_rerank=use_rerank)
        
        logger.info(f"\n📊 Query Analysis:")
        logger.info(f"  Colors detected: {explanation['parsed_query']['colors']}")
        logger.info(f"  Garments detected: {explanation['parsed_query']['garments']}")
        logger.info(f"  Scenes detected: {explanation['parsed_query']['scenes']}")
        logger.info(f"  Is compositional: {explanation['parsed_query']['is_compositional']}")
        
        logger.info(f"\n⚖️  Fusion Weights Used:")
        for component, weight in explanation['fusion_weights'].items():
            logger.info(f"  {component}: {weight:.3f}")
        
        logger.info(f"\n🎯 Top 10 Results:")
        for result in explanation['results']:
            img_name = Path(result['image_path']).name
            logger.info(f"\n  {result['rank']}. {img_name}")
            logger.info(f"     Score: {result['score']:.4f}")
            logger.info(f"     Colors: {', '.join(result['metadata']['colors'][:3])}")
            garment_names = [g['garment'] for g in result['metadata']['garments'][:2]]
            logger.info(f"     Garments: {', '.join(garment_names)}")
            scene_names = [s['scene'] for s in result['metadata']['scenes'][:2]]
            logger.info(f"     Scenes: {', '.join(scene_names)}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fashion Retrieval Pipeline')
    parser.add_argument('--skip-indexing', action='store_true',
                       help='Skip indexing step (use existing index)')
    
    args = parser.parse_args()
    
    if not args.skip_indexing:
        build_index()
    
    test_retrieval()
    
    logger.info("\n" + "="*60)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info("="*60)

if __name__ == "__main__":
    main()