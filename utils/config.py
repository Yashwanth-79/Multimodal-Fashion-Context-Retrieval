import os
from pathlib import Path
import torch

class Config:
    # Paths
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    LOGS_DIR = ROOT_DIR / "logs"
    MODELS_DIR = ROOT_DIR / "models"
    
    # Model configs
    CLIP_MODEL = "ViT-B/32"  
    EMBEDDING_DIM = {
        'clip': 512,
        'color': 64,
        'garment': 128,
        'scene': 128
    }
    TOTAL_DIM = sum(EMBEDDING_DIM.values())  # 832
    
    # Fusion weights (adaptive)
    DEFAULT_WEIGHTS = {
        'clip': 0.4,
        'color': 0.2,
        'garment': 0.2,
        'scene': 0.2
    }
    
    # Color vocabulary
    COLOR_VOCAB = [
        'red', 'blue', 'green', 'yellow', 'orange', 'purple', 
        'pink', 'brown', 'black', 'white', 'gray', 'beige',
        'navy', 'maroon', 'olive', 'teal', 'coral'
    ]
    
    # Garment types
    GARMENT_TYPES = [
    # Tops
    'shirt', 'button-down shirt', 'dress shirt', 'polo shirt',
    't-shirt', 'tee', 'blouse', 'tank top', 'crop top',
    'sweater', 'pullover', 'hoodie', 'sweatshirt',
    
    # Formal wear
    'suit', 'blazer', 'sport coat', 'vest', 'waistcoat',
    'tie', 'necktie', 'bow tie', 'cravat', 
    # Outerwear
    'jacket', 'coat', 'overcoat', 'trench coat',
    'parka', 'windbreaker', 'raincoat', 'cardigan',
    
    # Bottoms
    'pants', 'trousers', 'slacks', 'dress pants',
    'jeans', 'denim', 'chinos', 'khakis',
    'shorts', 'skirt', 'mini skirt', 'midi skirt', 'maxi skirt',
    'leggings', 'joggers',
    
    # Dresses
    'dress', 'gown', 'cocktail dress', 'evening dress',
    'sundress', 'maxi dress', 'mini dress',
    
    # Footwear
    'shoes', 'sneakers', 'trainers', 'running shoes',
    'boots', 'ankle boots', 'heels', 'high heels', 'pumps',
    'sandals', 'flip-flops', 'flats', 'loafers',
    
    # Accessories
    'hat', 'cap', 'beanie', 'fedora',
    'scarf', 'shawl', 'belt', 'gloves',
    'sunglasses', 'glasses', 'watch', 'jewelry',
    'bag', 'handbag', 'backpack', 'purse'
]
    
    # Scene types
    SCENE_TYPES = [
        'office', 'indoor', 'outdoor', 'street', 'park', 
        'urban', 'nature', 'runway', 'stage', 'home',
        'beach', 'mountain', 'city', 'forest'
    ]
    
    # Retrieval
    TOP_K = 10
    FAISS_INDEX_TYPE = "IVF" 
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()