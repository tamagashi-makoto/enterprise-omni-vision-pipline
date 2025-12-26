"""
Configuration for the Omni-Vision pipeline.
"""
from enum import Enum


class ModelType(str, Enum):
    """Enumeration of supported computer vision models."""
    YOLO_V12 = "YOLOv12"
    RF_DETR = "RF-DETR"
    FLORENCE_2 = "Florence-2"  # Replaced DINO-X
    SAM_3 = "SAM-3"


class Config:
    """Central configuration for the Omni-Vision pipeline."""
    
    # Thresholds
    DENSITY_THRESHOLD: int = 15
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # Model Latencies (estimated in seconds)
    LATENCY_YOLO: float = 0.05
    LATENCY_RF_DETR: float = 0.2
    LATENCY_FLORENCE_2: float = 0.3  # Replaced DINO-X
    LATENCY_SAM_3: float = 0.15

    # Paths
    MODEL_WEIGHTS_DIR: str = "src/weights/"
    
    # BoxSet / Pipeline Config
    MAX_MASK_BOXES: int = 20            # Top-K budget for SAM3 (prevent slowdown)
    NMS_IOU_THRESHOLD: float = 0.5      # NMS merge threshold for BoxSet
    MIN_BOX_AREA_RATIO: float = 0.0005  # Filter boxes smaller than this % of image area
    
    # Mode Settings
    MODE_DEFAULT: str = "auto"          # Default inference mode ("auto" | "query" | "smart_query")
    ENABLE_FLORENCE_RERANK: bool = False  # Enable Florence-2 reranking in query mode
    SAM3_TEXT_FIRST: bool = True        # Try SAM3 text-prompt first in query mode
    FALLBACK_IF_NO_MASK: bool = True    # Fall back to detector boxes if text fails
    
    # Gemma3 Query Generator Settings
    GEMMA3_ENABLED: bool = True         # Enable Gemma3 for smart query mode
    GEMMA3_MODEL_ID: str = "google/gemma-3-4b-it-qat-q4_0-gguf"  # HuggingFace model ID
    GEMMA3_MODEL_FILE: str = "gemma-3-4b-it-q4_0.gguf"  # GGUF file name
    GEMMA3_MAX_QUERIES: int = 5         # Maximum number of queries to generate
    GEMMA3_CONTEXT_SIZE: int = 2048     # Context window size
    GEMMA3_GPU_LAYERS: int = -1         # -1 = all layers on GPU
