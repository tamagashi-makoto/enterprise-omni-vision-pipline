import asyncio
from typing import List, Dict, Any, Optional, Tuple
from .config import Config, ModelType
from .model_wrappers import (
    YOLOv12Wrapper, 
    RFDETRWrapper, 
    DINOXWrapper, 
    SAM3Wrapper, 
    DetectionResult
)

class OmniVisionPipeline:
    """
    Intelligent orchestration pipeline that dynamically selects the best 
    computer vision model based on scene complexity and user intent.
    """

    def __init__(self):
        self.yolo = YOLOv12Wrapper()
        self.rf_detr = RFDETRWrapper()
        self.dino_x = DINOXWrapper()
        self.sam_3 = SAM3Wrapper()
        self.models_loaded = False

    async def load_models(self):
        """Initializes and loads all models (simulated)."""
        if not self.models_loaded:
            await asyncio.gather(
                self.yolo.load(),
                self.rf_detr.load(),
                self.dino_x.load(),
                self.sam_3.load()
            )
            self.models_loaded = True

    async def analyze(self, image: Any, text_query: Optional[str] = None) -> Dict[str, Any]:
        """
        Main pipeline entry point.
        
        Logic Flow:
        1. Stage 1 (Screening): Run YOLOv12.
        2. Stage 2 (Density Check): count > 15 -> Switch to RF-DETR.
        3. Stage 3 (Intent Check): if text_query -> Run DINO-X.
        4. Stage 4 (Segmentation): Run SAM 3 on final boxes.
        
        Args:
            image: Input image.
            text_query: Optional text prompt for identifying specific objects.
            
        Returns:
             JSON-compatible dictionary with metadata, detections, and segmentation status.
        """
        if not self.models_loaded:
            await self.load_models()

        active_mode = ModelType.YOLO_V12
        final_detections: List[DetectionResult] = []

        # --- Stage 1: Screening (YOLOv12) ---
        # Note: We always start with YOLO as a fast screener/default
        yolo_results = await self.yolo.predict(image)
        final_detections = yolo_results
        active_mode = "YOLO-Fast"

        # --- Stage 2: Density Check ---
        if len(yolo_results) > Config.DENSITY_THRESHOLD:
            # Too many objects, switch to high-precision model
            print(f"High density detected ({len(yolo_results)} objects). Switching to {ModelType.RF_DETR}.")
            rf_results = await self.rf_detr.predict(image)
            final_detections = rf_results
            active_mode = "RF-DETR-High-Res"
        else:
            print(f"Density normal ({len(yolo_results)} objects). Keeping {ModelType.YOLO_V12}.")

        # --- Stage 3: Intent Check ---
        # If user specifies what they are looking for, we prioritize that intent.
        # We assume DINO-X is best for open-vocabulary queries.
        if text_query:
            print(f"Text query received: '{text_query}'. Switching to {ModelType.DINO_X}.")
            dino_results = await self.dino_x.predict(image, text_query=text_query)
            final_detections = dino_results
            active_mode = f"DINO-X ({text_query})"

        # --- Stage 4: Segmentation (SAM 3) ---
        # Generate masks for whatever bounding boxes we decided on.
        boxes = [d.box for d in final_detections]
        masks = []
        segmentation_available = False
        
        if boxes:
            masks = await self.sam_3.predict(image, boxes=boxes)
            segmentation_available = True

        # Format Response
        response = {
            "meta": {
                "processing_mode": active_mode,
                "objects_detected": len(final_detections)
            },
            "detections": [d.to_dict() for d in final_detections],
            "masks_generated": len(masks), # Metadata only, actual masks might be huge blobs
            "segmentation_available": segmentation_available
        }
        
        return response
