"""
Omni-Vision Pipeline - Intelligent model orchestration with BoxSet merging.

Supports two inference modes:
- Mode "auto": YOLO → (RF-DETR if dense) → merge → filter → topK → SAM3 (box prompt)
- Mode "query": SAM3 text-first → (fallback to detectors if empty)
"""
import asyncio
import uuid
import base64
import io
from typing import List, Dict, Any, Optional, Literal, Union
from dataclasses import dataclass, field
from PIL import Image
import numpy as np

from .config import Config, ModelType
from .model_wrappers import (
    YOLOv12Wrapper, 
    RFDETRWrapper, 
    Florence2Wrapper,
    SAM3Wrapper, 
    DetectionResult,
    Gemma3QueryGenerator
)


# =============================================================================
# Mask Serialization Helpers
# =============================================================================

def encode_rle(mask: np.ndarray) -> Dict[str, Any]:
    """
    Encode a binary mask using Run-Length Encoding (RLE).
    
    Args:
        mask: HxW binary mask (bool or uint8 with 0/1 values)
        
    Returns:
        Dict with 'size', 'counts', and 'order' fields
    """
    h, w = mask.shape[:2]
    # Flatten in column-major (Fortran) order for COCO compatibility
    flat = mask.flatten(order='F').astype(np.uint8)
    
    # Compute run lengths
    counts = []
    current_val = 0  # Start counting zeros
    count = 0
    
    for val in flat:
        if val == current_val:
            count += 1
        else:
            counts.append(count)
            count = 1
            current_val = val
    counts.append(count)
    
    return {
        "size": [h, w],
        "counts": counts,
        "order": "F"
    }


def mask_to_png_base64(mask: np.ndarray) -> str:
    """
    Encode a binary mask as a base64 PNG string.
    
    Args:
        mask: HxW binary mask (bool or uint8)
        
    Returns:
        Base64-encoded PNG string
    """
    # Convert to uint8 (0 or 255)
    mask_uint8 = (mask.astype(np.uint8) * 255)
    
    # Create PIL Image
    img = Image.fromarray(mask_uint8, mode='L')
    
    # Encode to PNG in memory
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Base64 encode
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def serialize_masks(
    masks: List[np.ndarray], 
    mask_format: str = "rle"
) -> List[Union[Dict[str, Any], str, None]]:
    """
    Serialize masks to the requested format.
    
    Args:
        masks: List of HxW binary masks
        mask_format: "rle", "png_base64", or "none"
        
    Returns:
        List of serialized masks (RLE dicts, base64 strings, or empty for "none")
    """
    if mask_format == "none" or not masks:
        return []
    
    serialized = []
    for mask in masks:
        if mask is None:
            serialized.append(None)
            continue
            
        # Ensure mask is numpy array
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        elif hasattr(mask, 'numpy'):
            mask = mask.numpy()
        
        # Ensure 2D
        if mask.ndim > 2:
            mask = mask.squeeze()
        
        # Convert to binary
        mask = (mask > 0.5).astype(np.uint8)
        
        if mask_format == "rle":
            serialized.append(encode_rle(mask))
        elif mask_format == "png_base64":
            serialized.append(mask_to_png_base64(mask))
        else:
            serialized.append(None)
    
    return serialized


# =============================================================================
# BoxSet Data Structure
# =============================================================================

@dataclass
class BoxItem:
    """A single box candidate from any detector."""
    id: str
    box: List[float]  # [x1, y1, x2, y2]
    score: float
    label: Optional[str] = None
    source: Literal["yolo", "rfdetr", "florence"] = "yolo"
    meta: dict = field(default_factory=dict)

    def area(self) -> float:
        """Calculate box area."""
        return max(0, self.box[2] - self.box[0]) * max(0, self.box[3] - self.box[1])


# =============================================================================
# BoxSet Helper Functions
# =============================================================================

def _to_boxset_from_yolo(detections: List[DetectionResult]) -> List[BoxItem]:
    """Convert YOLO detections to BoxItems."""
    return [
        BoxItem(
            id=f"yolo_{i}",
            box=d.box,
            score=d.confidence,
            label=d.label,
            source="yolo"
        )
        for i, d in enumerate(detections)
    ]


def _to_boxset_from_rfdetr(detections: List[DetectionResult]) -> List[BoxItem]:
    """Convert RF-DETR detections to BoxItems."""
    return [
        BoxItem(
            id=f"rfdetr_{i}",
            box=d.box,
            score=d.confidence,
            label=d.label,
            source="rfdetr"
        )
        for i, d in enumerate(detections)
    ]


def _to_boxset_from_florence(detections: List[DetectionResult]) -> List[BoxItem]:
    """Convert Florence-2 detections to BoxItems."""
    return [
        BoxItem(
            id=f"florence_{i}",
            box=d.box,
            score=d.confidence,
            label=d.label,
            source="florence"
        )
        for i, d in enumerate(detections)
    ]


def _compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def _merge_boxsets_nms(boxes: List[BoxItem], iou_threshold: float) -> List[BoxItem]:
    """
    Merge overlapping boxes using NMS.
    Keeps the box with higher score when IoU > threshold.
    """
    if not boxes:
        return []
    
    # Sort by score descending
    sorted_boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
    kept = []
    
    for box in sorted_boxes:
        # Check if this box overlaps significantly with any kept box
        is_duplicate = False
        for kept_box in kept:
            if _compute_iou(box.box, kept_box.box) > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            kept.append(box)
    
    return kept


def _filter_boxes_by_area(
    boxes: List[BoxItem], 
    image_size: tuple, 
    min_area_ratio: float
) -> List[BoxItem]:
    """Remove boxes smaller than min_area_ratio of image area."""
    image_area = image_size[0] * image_size[1]
    min_area = image_area * min_area_ratio
    
    return [b for b in boxes if b.area() >= min_area]


def _select_top_k(boxes: List[BoxItem], k: int) -> List[BoxItem]:
    """Select top-K boxes by score."""
    sorted_boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
    return sorted_boxes[:k]


# =============================================================================
# Pipeline Class
# =============================================================================

class OmniVisionPipeline:
    """
    Intelligent orchestration pipeline that dynamically selects the best 
    computer vision model based on scene complexity and user intent.
    
    Uses BoxSet merging instead of switch-and-discard logic.
    SAM3 is the final source of truth for pixel-level masks.
    """

    def __init__(self):
        self.yolo = YOLOv12Wrapper()
        self.rf_detr = RFDETRWrapper()
        self.florence_2 = Florence2Wrapper()
        self.sam_3 = SAM3Wrapper()
        self.gemma3 = Gemma3QueryGenerator()
        self.models_loaded = False

    async def load_models(self):
        """Initializes and loads all models."""
        if not self.models_loaded:
            print("Loading all models...")
            await asyncio.gather(
                self.yolo.load(),
                self.rf_detr.load(),
                self.florence_2.load(),
                self.sam_3.load(),
                self.gemma3.load()
            )
            self.models_loaded = True
            print("All models loaded successfully!")

    async def analyze(
        self, 
        image: Image.Image, 
        text_query: Optional[str] = None,
        user_text: Optional[str] = None,
        mode: Optional[str] = None,
        mask_format: str = "rle"
    ) -> Dict[str, Any]:
        """
        Main pipeline entry point.
        
        Modes:
        - "auto" (default): Detector-based flow with SAM3 box prompts
        - "query": Text-query flow with SAM3 text-first, fallback to detectors
        - "smart_query": Gemma3 generates queries from natural language → SAM3 processes
        
        Args:
            image: PIL Image input.
            text_query: Optional text prompt for identifying specific objects (query mode).
            user_text: Optional natural language input for AI query generation (smart_query mode).
            mode: "auto" | "query" | "smart_query" (defaults to Config.MODE_DEFAULT)
            mask_format: "rle" (default), "png_base64", or "none"
            
        Returns:
            JSON-compatible dictionary with detections, masks, and metadata.
        """
        if not self.models_loaded:
            await self.load_models()
        
        # Determine mode
        effective_mode = mode or Config.MODE_DEFAULT
        if user_text and effective_mode in ("auto", None):
            # If user_text provided, switch to smart_query mode
            effective_mode = "smart_query"
        elif text_query and effective_mode == "auto":
            # If text_query provided but mode is auto, switch to query mode
            effective_mode = "query"
        
        image_size = (image.width, image.height)
        print(f"Pipeline: mode={effective_mode}, mask_format={mask_format}")
        
        if effective_mode == "smart_query":
            return await self._run_smart_query_mode(image, user_text, image_size, mask_format)
        elif effective_mode == "query":
            return await self._run_query_mode(image, text_query, image_size, mask_format)
        else:
            return await self._run_auto_mode(image, image_size, mask_format)

    async def _run_auto_mode(
        self, 
        image: Image.Image, 
        image_size: tuple,
        mask_format: str = "rle"
    ) -> Dict[str, Any]:
        """
        Auto mode: YOLO → (RF-DETR if dense) → merge → filter → topK → SAM3 (box prompt)
        
        Used for: E-commerce cutout, baseline privacy masking, closed-set inspection.
        """
        boxset: List[BoxItem] = []
        processing_stages = []
        
        # --- Stage 1: YOLO Screening ---
        print("Stage 1: Running YOLOv12 screening...")
        yolo_results = await self.yolo.predict(image)
        yolo_boxes = _to_boxset_from_yolo(yolo_results)
        boxset.extend(yolo_boxes)
        processing_stages.append(f"YOLO({len(yolo_results)})")
        
        # --- Stage 2: RF-DETR if high density ---
        if len(yolo_results) > Config.DENSITY_THRESHOLD:
            print(f"High density ({len(yolo_results)} objects). Adding RF-DETR results...")
            rf_results = await self.rf_detr.predict(image)
            rf_boxes = _to_boxset_from_rfdetr(rf_results)
            boxset.extend(rf_boxes)
            processing_stages.append(f"RF-DETR({len(rf_results)})")
        
        # --- Stage 3: Merge with NMS ---
        pre_nms_count = len(boxset)
        boxset = _merge_boxsets_nms(boxset, Config.NMS_IOU_THRESHOLD)
        print(f"NMS merge: {pre_nms_count} → {len(boxset)} boxes")
        
        # --- Stage 4: Filter by area ---
        pre_filter_count = len(boxset)
        boxset = _filter_boxes_by_area(boxset, image_size, Config.MIN_BOX_AREA_RATIO)
        if pre_filter_count != len(boxset):
            print(f"Area filter: {pre_filter_count} → {len(boxset)} boxes")
        
        # --- Stage 5: Select Top-K ---
        boxset = _select_top_k(boxset, Config.MAX_MASK_BOXES)
        print(f"Passing {len(boxset)} boxes to SAM3 (budget: {Config.MAX_MASK_BOXES})")
        
        # --- Stage 6: SAM3 Segmentation (box prompts) ---
        sam_result = await self._run_sam3_with_boxes(image, boxset)
        
        return self._format_response(
            boxset=boxset,
            sam_result=sam_result,
            mode="auto",
            processing_stages=processing_stages,
            mask_format=mask_format
        )

    async def _run_smart_query_mode(
        self, 
        image: Image.Image, 
        user_text: Optional[str],
        image_size: tuple,
        mask_format: str = "rle"
    ) -> Dict[str, Any]:
        """
        Smart query mode: Gemma3 generates queries → SAM3 processes each query
        
        Flow:
        1. Gemma3 analyzes user's natural language input
        2. Generates multiple relevant object detection queries
        3. Each query is processed by SAM3 for segmentation
        4. Results are merged and deduplicated
        
        Used for: Natural language object detection, intelligent search.
        """
        processing_stages = ["Gemma3"]
        all_masks = []
        all_scores = []
        all_boxes = []
        all_detections = []
        generated_queries = []
        
        if not user_text:
            # Fallback to auto mode if no user text
            return await self._run_auto_mode(image, image_size, mask_format)
        
        # --- Stage 1: Generate queries using Gemma3 ---
        print(f"Smart query mode: Analyzing user request: '{user_text}'")
        
        try:
            queries = await self.gemma3.generate_queries(user_text)
            generated_queries = queries
            print(f"Generated {len(queries)} queries: {queries}")
            processing_stages.append(f"Queries({len(queries)})")
        except Exception as e:
            print(f"Gemma3 query generation failed: {e}")
            # Fallback to using user_text directly as query
            queries = [user_text]
        
        if not queries:
            # No queries generated, fallback to auto mode
            print("No queries generated. Falling back to auto mode.")
            return await self._run_auto_mode(image, image_size, mask_format)
        
        # --- Stage 2: Process each query with SAM3 ---
        for query in queries:
            print(f"Processing query: '{query}'")
            
            try:
                # Try SAM3 text-prompted first
                if Config.SAM3_TEXT_FIRST and self.sam_3.enabled:
                    sam_result = await self.sam_3.predict(image, text_prompt=query)
                    masks = sam_result.get("masks", [])
                    
                    if masks and len(masks) > 0:
                        # Store results
                        for i, mask in enumerate(masks):
                            all_masks.append(mask)
                            score = sam_result.get("mask_scores", [0.9])[i] if i < len(sam_result.get("mask_scores", [])) else 0.9
                            all_scores.append(score)
                            box = sam_result.get("mask_boxes", [])[i] if i < len(sam_result.get("mask_boxes", [])) else [0, 0, 0, 0]
                            all_boxes.append(box if isinstance(box, list) else list(box))
                            all_detections.append({
                                "label": query,
                                "confidence": float(score),
                                "box": box if isinstance(box, list) else list(box),
                                "has_mask": True
                            })
                        processing_stages.append(f"SAM3({query}:{len(masks)})")
                        continue
                
                # Fallback: run query mode for this specific query
                result = await self._run_query_mode(image, query, image_size, "none")
                detections = result.get("detections", [])
                
                for det in detections:
                    all_detections.append({
                        **det,
                        "label": query  # Override with our query
                    })
                    
            except Exception as e:
                print(f"Query '{query}' processing failed: {e}")
                continue
        
        # --- Stage 3: Deduplicate and limit results ---
        # Remove duplicate boxes (by IoU threshold)
        if len(all_detections) > Config.MAX_MASK_BOXES:
            # Sort by confidence and take top K
            all_detections = sorted(all_detections, key=lambda d: d.get("confidence", 0), reverse=True)
            all_detections = all_detections[:Config.MAX_MASK_BOXES]
            
            # Also limit masks
            all_masks = all_masks[:Config.MAX_MASK_BOXES]
            all_scores = all_scores[:Config.MAX_MASK_BOXES]
            all_boxes = all_boxes[:Config.MAX_MASK_BOXES]
        
        # Serialize masks
        serialized_masks = serialize_masks(all_masks, mask_format)
        
        print(f"Smart query mode complete: {len(all_detections)} detections, {len(all_masks)} masks")
        
        return {
            "meta": {
                "processing_mode": " → ".join(processing_stages),
                "objects_detected": len(all_detections),
                "generated_queries": generated_queries
            },
            "detections": all_detections,
            "masks_generated": len(all_masks),
            "segmentation_available": len(all_masks) > 0,
            "masks": serialized_masks,
            "mask_scores": all_scores,
            "mask_boxes": all_boxes,
            "mode_used": "smart_query",
            "mask_format": mask_format,
            "queries_used": generated_queries
        }


    async def _run_query_mode(
        self, 
        image: Image.Image, 
        text_query: Optional[str],
        image_size: tuple,
        mask_format: str = "rle"
    ) -> Dict[str, Any]:
        """
        Query mode: SAM3 text-first → (fallback to detectors if empty)
        
        Used for: Open-vocabulary search, concept-based defect discovery.
        """
        processing_stages = []
        sam_result = None
        boxset: List[BoxItem] = []
        
        # --- Stage 1: Try SAM3 text-prompted first (if enabled) ---
        if text_query and Config.SAM3_TEXT_FIRST and self.sam_3.enabled:
            print(f"Query mode: Trying SAM3 text-prompt first: '{text_query}'")
            sam_result = await self.sam_3.predict(image, text_prompt=text_query)
            processing_stages.append("SAM3-text")
            
            masks = sam_result.get("masks", [])
            if masks and len(masks) > 0:
                print(f"SAM3 text-prompt found {len(masks)} masks")
                # Use mask boxes from SAM3 output
                mask_boxes = sam_result.get("mask_boxes", [])
                for i, box in enumerate(mask_boxes):
                    boxset.append(BoxItem(
                        id=f"sam3_text_{i}",
                        box=box if isinstance(box, list) else list(box),
                        score=sam_result.get("mask_scores", [0.9])[i] if i < len(sam_result.get("mask_scores", [])) else 0.9,
                        label=text_query,
                        source="florence"  # Using florence as source type for text-based
                    ))
                
                return self._format_response(
                    boxset=boxset,
                    sam_result=sam_result,
                    mode="query",
                    processing_stages=processing_stages,
                    mask_format=mask_format
                )
        
        # --- Stage 2: Fallback to detector-based flow ---
        if not Config.FALLBACK_IF_NO_MASK and sam_result:
            # Return empty result if fallback disabled
            return self._format_response(
                boxset=[],
                sam_result={"masks": [], "mask_scores": [], "mask_boxes": [], "metadata": {}},
                mode="query",
                processing_stages=processing_stages,
                mask_format=mask_format
            )
        
        print("SAM3 text-prompt empty or disabled. Falling back to detectors...")
        
        # Run YOLO
        yolo_results = await self.yolo.predict(image)
        boxset = _to_boxset_from_yolo(yolo_results)
        processing_stages.append(f"YOLO({len(yolo_results)})")
        
        # Add RF-DETR if dense
        if len(yolo_results) > Config.DENSITY_THRESHOLD:
            rf_results = await self.rf_detr.predict(image)
            boxset.extend(_to_boxset_from_rfdetr(rf_results))
            processing_stages.append(f"RF-DETR({len(rf_results)})")
        
        # --- Stage 3: Florence-2 rerank (optional) ---
        if Config.ENABLE_FLORENCE_RERANK and text_query:
            print(f"Running Florence-2 rerank with query: '{text_query}'")
            try:
                # Get Florence-2 detections for the text query
                florence_results = await self.florence_2.predict(image, text_query=text_query)
                if florence_results:
                    # Use Florence detections to rerank: prioritize boxes matching Florence output
                    florence_boxes = _to_boxset_from_florence(florence_results)
                    
                    # Boost scores of boxes that overlap with Florence results
                    for box in boxset:
                        for fb in florence_boxes:
                            if _compute_iou(box.box, fb.box) > 0.5:
                                box.score = min(1.0, box.score + 0.2)
                                box.meta["florence_match"] = True
                                break
                    
                    # Also add any unique Florence boxes
                    boxset.extend(florence_boxes)
                    processing_stages.append(f"Florence-2({len(florence_results)})")
            except Exception as e:
                print(f"Florence-2 rerank failed: {e}")
        
        # Merge, filter, select
        boxset = _merge_boxsets_nms(boxset, Config.NMS_IOU_THRESHOLD)
        boxset = _filter_boxes_by_area(boxset, image_size, Config.MIN_BOX_AREA_RATIO)
        boxset = _select_top_k(boxset, Config.MAX_MASK_BOXES)
        
        print(f"Passing {len(boxset)} boxes to SAM3 (budget: {Config.MAX_MASK_BOXES})")
        
        # --- Stage 4: SAM3 with box prompts ---
        sam_result = await self._run_sam3_with_boxes(image, boxset)
        
        return self._format_response(
            boxset=boxset,
            sam_result=sam_result,
            mode="query",
            processing_stages=processing_stages,
            mask_format=mask_format
        )

    async def _run_sam3_with_boxes(
        self, 
        image: Image.Image, 
        boxset: List[BoxItem]
    ) -> Dict[str, Any]:
        """Run SAM3 segmentation using box prompts."""
        if not boxset:
            return {"masks": [], "mask_scores": [], "mask_boxes": [], "metadata": {"source": "none"}}
        
        if not self.sam_3.enabled:
            print("SAM3 disabled. Returning empty masks.")
            return {"masks": [], "mask_scores": [], "mask_boxes": [], "metadata": {"source": "disabled"}}
        
        boxes = [b.box for b in boxset]
        print(f"Stage SAM3: Running segmentation on {len(boxes)} boxes...")
        
        return await self.sam_3.predict(image, boxes=boxes)

    def _format_response(
        self,
        boxset: List[BoxItem],
        sam_result: Dict[str, Any],
        mode: str,
        processing_stages: List[str],
        mask_format: str = "rle"
    ) -> Dict[str, Any]:
        """Format the final API response with serialized masks."""
        # Convert BoxItems to detection dicts
        raw_masks = sam_result.get("masks", [])
        has_masks = raw_masks is not None and len(raw_masks) > 0 if hasattr(raw_masks, '__len__') else False
        
        detections = [
            {
                "label": b.label or "object",
                "confidence": b.score,
                "box": b.box,
                "has_mask": has_masks
            }
            for b in boxset
        ]
        
        # Enforce MAX_MASK_BOXES on output masks
        mask_scores = sam_result.get("mask_scores", [])
        mask_boxes = sam_result.get("mask_boxes", [])
        
        # Prune if too many masks (by score then area)
        if len(raw_masks) > Config.MAX_MASK_BOXES:
            print(f"Pruning masks: {len(raw_masks)} → {Config.MAX_MASK_BOXES}")
            # Sort by score descending
            if mask_scores:
                indices = sorted(range(len(mask_scores)), key=lambda i: mask_scores[i], reverse=True)
            else:
                indices = list(range(len(raw_masks)))
            indices = indices[:Config.MAX_MASK_BOXES]
            raw_masks = [raw_masks[i] for i in indices]
            mask_scores = [mask_scores[i] for i in indices] if mask_scores else []
            mask_boxes = [mask_boxes[i] for i in indices] if mask_boxes else []
        
        # Serialize masks
        serialized_masks = serialize_masks(raw_masks, mask_format)
        masks_count = len(raw_masks) if hasattr(raw_masks, '__len__') else 0
        print(f"Returning {masks_count} masks (format: {mask_format})")
        
        return {
            "meta": {
                "processing_mode": " → ".join(processing_stages) if processing_stages else mode,
                "objects_detected": len(detections)
            },
            "detections": detections,
            "masks_generated": masks_count,
            "segmentation_available": masks_count > 0,
            "masks": serialized_masks,
            "mask_scores": mask_scores,
            "mask_boxes": mask_boxes,
            "mode_used": mode,
            "mask_format": mask_format
        }
