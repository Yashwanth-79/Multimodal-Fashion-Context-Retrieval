import numpy as np
import faiss
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
import sys
sys.path.append('..')

from indexer.feature_extractor import MultimodalFeatureExtractor
from utils.config import config
from utils.logger import logger

class IndexBuilder:
    """
    Build FAISS index from image dataset
    """
    
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else config.RAW_DATA_DIR
        self.extractor = MultimodalFeatureExtractor()
        
        self.embeddings = []
        self.metadata_list = []
        self.image_paths = []
        
    def collect_images(self) -> List[Path]:
        """Collect all image paths from data directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        images = []
        
        for ext in image_extensions:
            images.extend(self.data_dir.glob(f"**/*{ext}"))
        
        logger.info(f"✓ Found {len(images)} images")
        return images
    
    def process_images(self, image_paths: List[Path] = None, 
                      batch_size: int = 100):
        """
        Process all images and extract features
        """
        if image_paths is None:
            image_paths = self.collect_images()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {len(image_paths)} images...")
        logger.info(f"{'='*60}\n")
        
        for i, img_path in enumerate(tqdm(image_paths, desc="Extracting features")):
            try:
                # Extract features
                combined_emb, metadata, _ = self.extractor.extract_and_combine(img_path)
                
                self.embeddings.append(combined_emb)
                self.metadata_list.append(metadata)
                self.image_paths.append(str(img_path))
                
                # Periodic save (in case of crashes)
                if (i + 1) % batch_size == 0:
                    self._save_checkpoint(i + 1)
                    
            except Exception as e:
                logger.error(f"\n Error processing {img_path}: {e}")
                continue
        
        logger.info(f"\n✓ Successfully processed {len(self.embeddings)} images")
    
    def build_faiss_index(self, index_type: str = "IVF"):
        """
        Build FAISS index from embeddings
        
        Args:
            index_type: "Flat" (exact) or "IVF" (approximate, faster for large datasets)
        """
        if not self.embeddings:
            raise ValueError("No embeddings to index! Run process_images() first.")
        
        embeddings_array = np.array(self.embeddings).astype('float32')
        d = embeddings_array.shape[1]  # Dimension
        
        logger.info(f"\nBuilding FAISS index...")
        logger.info(f"  Index type: {index_type}")
        logger.info(f"  Embeddings shape: {embeddings_array.shape}")
        
        if index_type == "Flat":
            index = faiss.IndexFlatIP(d)  # Inner product (cosine similarity)
            
        elif index_type == "IVF":
            nlist = min(100, len(self.embeddings) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
            
            # Train index
            logger.info("  Training index...")
            index.train(embeddings_array)
            index.nprobe = 10  # Number of clusters to search
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add embeddings
        logger.info("  Adding embeddings to index...")
        index.add(embeddings_array)
        
        logger.info(f"✓ Index built successfully! Total vectors: {index.ntotal}")
        
        return index
    
    def save_index(self, index, output_dir: str = None):
        """
        Save FAISS index and metadata
        """
        if output_dir is None:
            output_dir = config.PROCESSED_DATA_DIR
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = output_dir / "faiss_index.bin"
        faiss.write_index(index, str(index_path))
        logger.info(f"✓ Saved FAISS index to {index_path}")
        
        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'image_paths': self.image_paths,
                'metadata': self.metadata_list
            }, f, indent=2)
        logger.info(f"✓ Saved metadata to {metadata_path}")
        
        # Save embeddings (for later analysis)
        embeddings_path = output_dir / "embeddings.npy"
        np.save(embeddings_path, np.array(self.embeddings))
        logger.info(f"✓ Saved embeddings to {embeddings_path}")
    
    def _save_checkpoint(self, num_processed: int):
        """Save intermediate checkpoint"""
        checkpoint_dir = config.PROCESSED_DATA_DIR / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'embeddings': self.embeddings,
            'metadata': self.metadata_list,
            'image_paths': self.image_paths,
            'num_processed': num_processed
        }
        
        checkpoint_path = checkpoint_dir / f"checkpoint_{num_processed}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def build_complete_index(self, index_type: str = "Flat", output_dir: str = None):
        """
        Complete pipeline: collect, process, index, save
        """
        # Step 1: Collect images
        image_paths = self.collect_images()
        
        # Step 2: Process images
        self.process_images(image_paths)
        
        # Step 3: Build index
        index = self.build_faiss_index(index_type=index_type)
        
        # Step 4: Save everything
        self.save_index(index, output_dir)
        
        logger.info(f"\n{'='*60}")
        logger.info("✓ INDEX BUILD COMPLETE!")
        logger.info(f"{'='*60}")


# Main execution
# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build fashion retrieval index')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for index')
    parser.add_argument('--index_type', type=str, default='Flat',
                       choices=['Flat', 'IVF'],
                       help='FAISS index type')
    
    args = parser.parse_args()
    
    # Build index
    builder = IndexBuilder(data_dir=args.data_dir)
    builder.build_complete_index(
        index_type=args.index_type,
        output_dir=args.output_dir
    )