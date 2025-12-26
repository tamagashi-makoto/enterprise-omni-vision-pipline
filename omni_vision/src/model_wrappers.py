"""
Model wrappers for the Omni-Vision pipeline.
Implements real model inference with GPU support.
"""
import os
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import torch
from PIL import Image
import numpy as np

from .config import Config, ModelType

# Weight file paths
WEIGHTS_DIR = Path(__file__).parent / "weights"


@dataclass
class DetectionResult:
    """Standardized output for detection models."""
    label: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2]
    mask: Optional[np.ndarray] = None  # Optional segmentation mask

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "label": self.label,
            "confidence": self.confidence,
            "box": self.box
        }
        if self.mask is not None:
            result["has_mask"] = True
        return result


class ModelWrapper(ABC):
    """Abstract base class for all model wrappers."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
    
    @abstractmethod
    async def load(self):
        """Load the model weights."""
        pass

    @abstractmethod
    async def predict(self, image: Any, **kwargs) -> Any:
        """Run inference on the image."""
        pass


class YOLOv12Wrapper(ModelWrapper):
    """
    Wrapper for YOLOv12 (Screening Model).
    Uses ultralytics library with pretrained weights in src/weights/.
    """

    async def load(self):
        """Load YOLO model for object detection."""
        from ultralytics import YOLO
        
        print(f"Loading {ModelType.YOLO_V12}... Device: {self.device}")
        
        # YOLOv12 weights have ultralytics version compatibility issues (AAttn.qkv attribute error)
        # Directly use YOLOv8m which is stable and well-tested
        try:
            # Try YOLOv8m first (download if not cached)
            print("Loading YOLOv8m model...")
            self.model = YOLO("yolov8m.pt")
            print("Loaded YOLOv8m successfully")
        except Exception as e:
            print(f"YOLOv8m failed: {e}, trying YOLOv8s...")
            self.model = YOLO("yolov8s.pt")
            print("Loaded YOLOv8s successfully")
        
        # Move to GPU if available
        self.model.to(self.device)
        print(f"YOLO ready on {self.device}")

    async def predict(self, image: Image.Image, **kwargs) -> List[DetectionResult]:
        """
        Run YOLOv12 object detection.
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            List of detected objects with bounding boxes.
        """
        if self.model is None:
            await self.load()
        
        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, 
            lambda: self.model.predict(image, verbose=False)
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                label = result.names[cls]
                
                detections.append(DetectionResult(
                    label=label,
                    confidence=conf,
                    box=box
                ))
        
        return detections


class RFDETRWrapper(ModelWrapper):
    """
    Wrapper for RF-DETR using the official rfdetr package.
    Runs locally on GPU for high-precision object detection.
    """

    def __init__(self):
        super().__init__()

    async def load(self):
        """Load RF-DETR model."""
        from rfdetr import RFDETRBase
        
        print(f"Loading {ModelType.RF_DETR}... Device: {self.device}")
        
        self.model = RFDETRBase()
        
        # Note: optimize_for_inference() can cause TorchScript tracing errors
        # Skipping for now - inference still works without optimization
        # if self.device == "cuda":
        #     self.model.optimize_for_inference()
        
        print(f"RF-DETR ready on {self.device}")


    async def predict(self, image: Image.Image, threshold: float = 0.5, **kwargs) -> List[DetectionResult]:
        """
        Run RF-DETR detection.
        
        Args:
            image: PIL Image
            threshold: Confidence threshold (default: 0.5)
        
        Returns:
            List of high-precision detections.
        """
        if self.model is None:
            await self.load()
        
        # Import COCO classes for labels
        from rfdetr.util.coco_classes import COCO_CLASSES
        
        # Run inference in thread pool
        loop = asyncio.get_event_loop()
        
        def _inference():
            return self.model.predict(image, threshold=threshold)
        
        result = await loop.run_in_executor(None, _inference)
        
        detections = []
        for i, (class_id, conf) in enumerate(zip(result.class_id, result.confidence)):
            box = result.xyxy[i].tolist()  # [x1, y1, x2, y2]
            label = COCO_CLASSES[int(class_id)]
            
            detections.append(DetectionResult(
                label=label,
                confidence=float(conf),
                box=box
            ))
        
        return detections


class Florence2Wrapper(ModelWrapper):
    """
    Wrapper for Florence-2 (Open-Vocabulary Detection).
    Uses Microsoft's Florence-2 model from HuggingFace.
    Replaces DINO-X as API key is not available.
    """

    def __init__(self):
        super().__init__()
        self.processor = None

    async def load(self):
        """Load Florence-2 model from HuggingFace."""
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        print(f"Loading {ModelType.FLORENCE_2}... Device: {self.device}")
        
        if self.device == "cpu":
            print("WARNING: Florence-2 on CPU will be slow. GPU recommended.")
        
        model_id = "microsoft/Florence-2-base"
        
        self.processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            attn_implementation="eager"  # Fix for transformers compatibility
        ).to(self.device)
        
        print(f"Florence-2 loaded on {self.device}")


    async def predict(self, image: Image.Image, text_query: str = "", **kwargs) -> List[DetectionResult]:
        """
        Run Florence-2 open-vocabulary detection.
        
        Args:
            image: PIL Image
            text_query: Text prompt describing what to detect
        
        Returns:
            List of detections matching the query.
        """
        if self.model is None:
            await self.load()
        
        if not text_query:
            return []
        
        # Florence-2 task for object detection with text
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        prompt = f"{task_prompt} {text_query}"
        
        loop = asyncio.get_event_loop()
        
        def _inference():
            inputs = self.processor(
                text=prompt, 
                images=image, 
                return_tensors="pt"
            ).to(self.device)
            
            # Convert pixel_values to the model's dtype (float16 on CUDA)
            if self.device == "cuda":
                inputs["pixel_values"] = inputs["pixel_values"].half()
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    do_sample=False,
                    num_beams=1  # Use greedy decoding to avoid beam search issues
                )
            
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=False
            )[0]
            
            parsed = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.width, image.height)
            )
            return parsed
        
        result = await loop.run_in_executor(None, _inference)
        
        detections = []
        if task_prompt in result:
            data = result[task_prompt]
            boxes = data.get("bboxes", [])
            labels = data.get("labels", [])
            
            for box, label in zip(boxes, labels):
                detections.append(DetectionResult(
                    label=label,
                    confidence=0.9,  # Florence-2 doesn't output confidence
                    box=list(box)
                ))
        
        return detections


class SAM3Wrapper(ModelWrapper):
    """
    Wrapper for SAM 3 (Segment Anything Model 3).
    Supports:
    - Text-prompted segmentation via SAM3 text API
    - Box-prompted segmentation via center-point conversion (workaround)
    
    Note: SAM3 works best on GPU. CPU execution is disabled gracefully.
    """

    def __init__(self):
        super().__init__()
        self.processor = None
        self.enabled = True  # Set to False if GPU unavailable

    async def load(self):
        """Load SAM3 model onto GPU. Gracefully disables on CPU."""
        print(f"Loading {ModelType.SAM_3}... Device: {self.device}")
        
        if self.device == "cpu":
            print("WARNING: SAM3 requires GPU. Disabling SAM3 (pipeline will degrade gracefully).")
            self.enabled = False
            return
        
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            # HuggingFace auth from environment
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                from huggingface_hub import login
                login(token=hf_token)
            
            # Build SAM3 model
            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model)
            self.enabled = True
            
            print(f"SAM3 loaded successfully on {self.device}")
        except Exception as e:
            print(f"WARNING: Failed to load SAM3: {e}. Disabling SAM3.")
            self.enabled = False

    async def predict(
        self, 
        image: Image.Image, 
        boxes: Optional[List[List[float]]] = None,
        text_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate segmentation masks using SAM3.
        
        Supports two modes:
        1. text_prompt provided: Use SAM3's text-prompted segmentation
        2. boxes provided: Use multi-point prompts per box (robust fallback)
        
        Note on box prompts:
        Ultralytics SAM3 does not expose native box prompts in some environments.
        We emulate box prompting via multi-point prompts (5 points per box):
        - Center point
        - 4 corner points inset by 10% of box dimensions
        Each box is processed sequentially to avoid prompt mixing in dense scenes.
        
        Args:
            image: PIL Image
            boxes: Optional bounding boxes [[x1,y1,x2,y2], ...] for segmentation
            text_prompt: Optional text prompt for segmentation
            
        Returns:
            Standardized dict with masks, mask_scores, mask_boxes, metadata
        """
        # Return empty result if SAM3 is disabled
        if not self.enabled:
            return self._empty_result("disabled")
        
        if self.model is None:
            await self.load()
            if not self.enabled:
                return self._empty_result("disabled")
        
        loop = asyncio.get_event_loop()
        
        if text_prompt:
            # Text-prompted segmentation
            result = await loop.run_in_executor(None, lambda: self._run_text_prompt(image, text_prompt))
            return self._standardize_output(result, [], "text")
        elif boxes and len(boxes) > 0:
            # Box-prompted segmentation via multi-point prompts
            result = await loop.run_in_executor(None, lambda: self._run_box_prompts(image, boxes))
            return result
        else:
            return self._empty_result("no_prompt")
    
    def _run_text_prompt(self, image: Image.Image, text_prompt: str) -> Dict[str, Any]:
        """Run text-prompted segmentation."""
        try:
            inference_state = self.processor.set_image(image)
            output = self.processor.set_text_prompt(
                state=inference_state, 
                prompt=text_prompt
            )
            return output if isinstance(output, dict) else {"masks": [], "scores": [], "boxes": []}
        except Exception as e:
            print(f"SAM3 text prompt error: {e}")
            return {"masks": [], "scores": [], "boxes": []}
    
    def _run_box_prompts(self, image: Image.Image, boxes: List[List[float]]) -> Dict[str, Any]:
        """
        Run box-prompted segmentation using multi-point prompts per box.
        
        For each box, generates 5 positive points:
        - Center point
        - 4 corner points inset by 10% of box dimensions
        
        Processes boxes sequentially to avoid prompt mixing.
        """
        all_masks = []
        all_scores = []
        all_boxes = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
            
            # Generate multi-point prompts for this box
            inset_x = w * 0.1
            inset_y = h * 0.1
            
            points = [
                [(x1 + x2) / 2, (y1 + y2) / 2],  # Center
                [x1 + inset_x, y1 + inset_y],    # Top-left inset
                [x2 - inset_x, y1 + inset_y],    # Top-right inset
                [x1 + inset_x, y2 - inset_y],    # Bottom-left inset
                [x2 - inset_x, y2 - inset_y],    # Bottom-right inset
            ]
            labels = [1, 1, 1, 1, 1]  # All positive (foreground)
            
            try:
                inference_state = self.processor.set_image(image)
                
                # Try point prompt API
                if hasattr(self.processor, 'set_point_prompt'):
                    output = self.processor.set_point_prompt(
                        state=inference_state,
                        points=points,
                        labels=labels
                    )
                elif hasattr(self.processor, 'add_points'):
                    # Alternative API
                    output = self.processor.add_points(
                        state=inference_state,
                        points=points,
                        labels=labels
                    )
                else:
                    # Last resort: single center point
                    center = points[0]
                    output = self.processor.set_text_prompt(
                        state=inference_state,
                        prompt="object"
                    )
                
                if not isinstance(output, dict):
                    continue
                
                masks = output.get("masks", [])
                scores = output.get("scores", [])
                
                # Select best mask for this box (highest score or largest area)
                if masks is not None and len(masks) > 0:
                    best_idx = 0
                    if scores is not None and len(scores) > 0:
                        if hasattr(scores, 'tolist'):
                            scores_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
                        else:
                            scores_list = list(scores) if not isinstance(scores, list) else scores
                        if scores_list:
                            best_idx = max(range(len(scores_list)), key=lambda i: scores_list[i])
                    
                    # Get the best mask
                    if hasattr(masks, '__getitem__'):
                        best_mask = masks[best_idx]
                        if hasattr(best_mask, 'cpu'):
                            best_mask = best_mask.cpu().numpy()
                        elif hasattr(best_mask, 'numpy'):
                            best_mask = best_mask.numpy()
                        
                        all_masks.append(best_mask)
                        best_score = scores_list[best_idx] if scores_list and best_idx < len(scores_list) else 0.9
                        all_scores.append(float(best_score))
                        all_boxes.append(box)
                        
            except Exception as e:
                print(f"SAM3 box prompt error for box {box}: {e}")
                continue
        
        return {
            "masks": all_masks,
            "mask_scores": all_scores,
            "mask_boxes": all_boxes,
            "metadata": {"source": "box_prompt", "mask_count": len(all_masks)}
        }
    
    def _empty_result(self, reason: str) -> Dict[str, Any]:
        """Return empty standardized result."""
        return {
            "masks": [],
            "mask_scores": [],
            "mask_boxes": [],
            "metadata": {"source": "none", "reason": reason}
        }
    
    def _standardize_output(
        self, 
        result: Dict[str, Any], 
        input_boxes: List[List[float]],
        source: str
    ) -> Dict[str, Any]:
        """Standardize SAM3 output to consistent format."""
        if not isinstance(result, dict):
            return self._empty_result("invalid_output")
        
        masks = result.get("masks", [])
        scores = result.get("scores", [])
        boxes = result.get("boxes", input_boxes)
        
        # Convert masks to list if needed
        if hasattr(masks, 'cpu'):
            masks = [m.cpu().numpy() for m in masks]
        elif isinstance(masks, np.ndarray):
            masks = [masks] if masks.ndim == 2 else list(masks)
        
        # Ensure scores is a list of floats
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        if not isinstance(scores, list):
            scores = [float(scores)] if scores else []
        
        # Ensure boxes is a list of lists
        if hasattr(boxes, 'tolist'):
            boxes = boxes.tolist()
        
        return {
            "masks": masks,
            "mask_scores": scores,
            "mask_boxes": boxes,
            "metadata": {"source": source, "mask_count": len(masks)}
        }


class Gemma3QueryGenerator(ModelWrapper):
    """
    Gemma3-4b-it GGUF model for generating search queries from user text.
    
    Uses llama-cpp-python to load the quantized GGUF model from HuggingFace.
    Converts natural language user input into multiple object detection queries.
    """

    def __init__(self):
        super().__init__()
        self.enabled = True

    async def load(self):
        """Load Gemma3 GGUF model via llama-cpp-python from HuggingFace."""
        print(f"Loading Gemma3 Query Generator... Device: {self.device}")
        
        if not Config.GEMMA3_ENABLED:
            print("Gemma3 is disabled in config. Skipping load.")
            self.enabled = False
            return
        
        try:
            from llama_cpp import Llama
            
            # Load model directly from HuggingFace
            # llama-cpp-python supports hf:// prefix for HuggingFace models
            hf_token = os.environ.get("HF_TOKEN")
            
            print(f"Downloading/loading model from HuggingFace: {Config.GEMMA3_MODEL_ID}")
            
            # Determine GPU layers
            n_gpu_layers = Config.GEMMA3_GPU_LAYERS if self.device == "cuda" else 0
            
            self.model = Llama.from_pretrained(
                repo_id=Config.GEMMA3_MODEL_ID,
                filename=Config.GEMMA3_MODEL_FILE,
                n_ctx=Config.GEMMA3_CONTEXT_SIZE,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            
            self.enabled = True
            print(f"Gemma3 loaded successfully (GPU layers: {n_gpu_layers})")
            
        except ImportError as e:
            print(f"WARNING: llama-cpp-python not installed: {e}")
            print("Install with: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python")
            self.enabled = False
        except Exception as e:
            print(f"WARNING: Failed to load Gemma3: {e}")
            self.enabled = False

    async def predict(self, image: Any, **kwargs) -> Any:
        """Not used - Gemma3 is for text generation only."""
        raise NotImplementedError("Gemma3 is for query generation, not image prediction")

    async def generate_queries(
        self, 
        user_text: str, 
        max_queries: int = None
    ) -> List[str]:
        """
        Generate multiple search queries from user's natural language input.
        
        Args:
            user_text: User's natural language description (e.g., "この画像の中の車を見つけて")
            max_queries: Maximum number of queries to generate (default: Config.GEMMA3_MAX_QUERIES)
            
        Returns:
            List of search query strings (e.g., ["car", "vehicle", "automobile"])
        """
        if not self.enabled:
            # Fallback: extract simple nouns from user text
            return self._fallback_extract_queries(user_text)
        
        if self.model is None:
            await self.load()
            if not self.enabled:
                return self._fallback_extract_queries(user_text)
        
        max_queries = max_queries or Config.GEMMA3_MAX_QUERIES
        
        # Construct the prompt for query generation
        system_prompt = """You are a query generator for an object detection system.
Given a user's request in natural language, extract and generate relevant object detection queries.
Output ONLY a JSON array of simple object names (single words or short phrases) that can be detected in an image.
Do not include explanations, just the JSON array.

Example input: "Find all vehicles and people in this image"
Example output: ["car", "truck", "bus", "person", "pedestrian"]

Example input: "この画像の中の動物を探して"
Example output: ["dog", "cat", "bird", "horse", "animal"]

Example input: "Detect manufacturing defects"
Example output: ["scratch", "dent", "crack", "defect", "damage"]"""

        user_prompt = f"Generate {max_queries} or fewer object detection queries for: \"{user_text}\""
        
        # Format for Gemma3 instruction-tuned model
        prompt = f"""<start_of_turn>user
{system_prompt}

{user_prompt}<end_of_turn>
<start_of_turn>model
"""
        
        loop = asyncio.get_event_loop()
        
        def _generate():
            try:
                output = self.model(
                    prompt,
                    max_tokens=256,
                    temperature=0.3,
                    stop=["<end_of_turn>", "\n\n"],
                    echo=False
                )
                return output["choices"][0]["text"].strip()
            except Exception as e:
                print(f"Gemma3 generation error: {e}")
                return "[]"
        
        response = await loop.run_in_executor(None, _generate)
        
        # Parse the JSON response
        queries = self._parse_query_response(response, max_queries)
        
        print(f"Gemma3 generated queries: {queries}")
        return queries

    def _parse_query_response(self, response: str, max_queries: int) -> List[str]:
        """Parse the model's JSON response into a list of queries."""
        import json
        import re
        
        try:
            # Try to extract JSON array from response
            # Handle cases where the model adds extra text
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                queries = json.loads(json_match.group())
                if isinstance(queries, list):
                    # Filter and clean queries
                    cleaned = []
                    for q in queries:
                        if isinstance(q, str) and q.strip():
                            cleaned.append(q.strip().lower())
                    return cleaned[:max_queries]
        except (json.JSONDecodeError, Exception) as e:
            print(f"Failed to parse Gemma3 response as JSON: {e}")
        
        # Fallback: try to extract words from response
        words = re.findall(r'"([^"]+)"', response)
        if words:
            return [w.lower().strip() for w in words[:max_queries]]
        
        return []

    def _fallback_extract_queries(self, user_text: str) -> List[str]:
        """
        Simple fallback when Gemma3 is not available.
        Extracts potential object names from user text using basic patterns.
        """
        import re
        
        # Common object keywords to look for
        common_objects = [
            "car", "person", "dog", "cat", "bird", "truck", "bus", "bicycle",
            "motorcycle", "airplane", "boat", "chair", "table", "phone", "laptop",
            "bottle", "cup", "food", "plant", "animal", "vehicle", "face",
            "license plate", "defect", "scratch", "damage"
        ]
        
        # Japanese to English mapping for common terms
        ja_to_en = {
            "車": "car", "人": "person", "犬": "dog", "猫": "cat", "鳥": "bird",
            "トラック": "truck", "バス": "bus", "自転車": "bicycle", "バイク": "motorcycle",
            "飛行機": "airplane", "船": "boat", "椅子": "chair", "テーブル": "table",
            "動物": "animal", "乗り物": "vehicle", "顔": "face", "食べ物": "food",
            "植物": "plant", "傷": "scratch", "欠陥": "defect", "ダメージ": "damage"
        }
        
        queries = []
        text_lower = user_text.lower()
        
        # Check for common English objects
        for obj in common_objects:
            if obj in text_lower:
                queries.append(obj)
        
        # Check for Japanese terms
        for ja, en in ja_to_en.items():
            if ja in user_text:
                if en not in queries:
                    queries.append(en)
        
        # If no matches found, try to extract nouns (simple heuristic)
        if not queries:
            # Extract quoted strings or single words
            words = re.findall(r'"([^"]+)"|\'([^\']+)\'|(\b\w+\b)', user_text)
            for match in words:
                word = next((w for w in match if w), None)
                if word and len(word) > 2 and word.lower() not in ["the", "and", "for", "find", "detect", "this", "image"]:
                    queries.append(word.lower())
        
        return queries[:Config.GEMMA3_MAX_QUERIES] if queries else ["object"]

