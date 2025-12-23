from enum import Enum

class ModelType(str, Enum):
    """Enumeration of supported computer vision models."""
    YOLO_V12 = "YOLOv12"
    RF_DETR = "RF-DETR"
    DINO_X = "DINO-X"
    SAM_3 = "SAM-3"

class Config:
    """Central configuration for the Omni-Vision pipeline."""
    
    # Thresholds
    DENSITY_THRESHOLD: int = 15
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # Model Latencies (Simulated in seconds)
    LATENCY_YOLO: float = 0.05
    LATENCY_RF_DETR: float = 0.2
    LATENCY_DINO_X: float = 0.1
    LATENCY_SAM_3: float = 0.15

    # Paths (Placeholder)
    MODEL_WEIGHTS_DIR: str = "weights/"
