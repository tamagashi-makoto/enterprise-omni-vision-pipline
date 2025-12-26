#!/usr/bin/env python3
"""
Demo script for Omni-Vision Pipeline.

Downloads a sample image and runs the pipeline to show how it works.
Usage: python demo.py
"""
import asyncio
import sys
import os
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image
import numpy as np
import urllib.request
import io


async def main():
    """Run pipeline demo with a sample image."""
    print("=" * 60)
    print("Omni-Vision Pipeline Demo")
    print("=" * 60)
    
    # --- Step 1: Get a sample image ---
    print("\n[1/4] Loading sample image...")
    
    # Use a sample image from the web (a street scene with cars/people)
    sample_url = "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=640"
    
    try:
        # Try downloading
        print(f"    Downloading from: {sample_url[:50]}...")
        with urllib.request.urlopen(sample_url, timeout=10) as response:
            image_data = response.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        print(f"    ✓ Downloaded image: {image.width}x{image.height}")
    except Exception as e:
        print(f"    ✗ Download failed: {e}")
        print("    Creating synthetic test image instead...")
        # Create a synthetic image with colored rectangles to simulate objects
        image = Image.new('RGB', (640, 480), color=(200, 200, 200))
        # Add some colored rectangles to simulate objects
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        draw.rectangle([50, 100, 200, 300], fill=(255, 0, 0))    # Red "object"
        draw.rectangle([300, 150, 500, 400], fill=(0, 0, 255))   # Blue "object"
        draw.rectangle([450, 50, 600, 200], fill=(0, 255, 0))    # Green "object"
        print(f"    ✓ Created synthetic image: {image.width}x{image.height}")
    
    # Save for reference
    demo_image_path = Path(__file__).parent / "demo_input.jpg"
    image.save(demo_image_path)
    print(f"    Saved to: {demo_image_path}")
    
    # --- Step 2: Import and initialize pipeline ---
    print("\n[2/4] Initializing pipeline...")
    
    try:
        from src.pipeline import OmniVisionPipeline
        from src.config import Config
        
        pipeline = OmniVisionPipeline()
        print("    ✓ Pipeline created")
        
        print("    Loading models (this may take a while on first run)...")
        await pipeline.load_models()
        print("    ✓ All models loaded")
        
    except Exception as e:
        print(f"    ✗ Pipeline initialization failed: {e}")
        print("\n    This may be because required dependencies are not installed.")
        print("    Please ensure you have run: pip install -r requirements.txt")
        return
    
    # --- Step 3: Run pipeline in AUTO mode ---
    print("\n[3/4] Running pipeline (mode=auto)...")
    print("-" * 60)
    
    try:
        result = await pipeline.analyze(
            image=image,
            mode="auto",
            mask_format="rle"
        )
        
        print("\n📊 AUTO MODE RESULTS:")
        print(f"    Mode used: {result.get('mode_used')}")
        print(f"    Processing: {result.get('meta', {}).get('processing_mode')}")
        print(f"    Objects detected: {result.get('meta', {}).get('objects_detected')}")
        print(f"    Masks generated: {result.get('masks_generated')}")
        print(f"    Mask format: {result.get('mask_format')}")
        
        # Show detections
        detections = result.get('detections', [])
        if detections:
            print(f"\n    Detections ({len(detections)}):")
            for i, det in enumerate(detections[:5]):  # Show first 5
                print(f"      [{i+1}] {det['label']}: {det['confidence']:.2f} @ {[int(x) for x in det['box']]}")
            if len(detections) > 5:
                print(f"      ... and {len(detections)-5} more")
        
        # Show mask info
        masks = result.get('masks', [])
        if masks:
            print(f"\n    Masks ({len(masks)}):")
            for i, mask in enumerate(masks[:3]):  # Show first 3
                if isinstance(mask, dict):  # RLE
                    print(f"      [{i+1}] RLE: size={mask.get('size')}, counts_len={len(mask.get('counts', []))}")
                else:  # base64
                    print(f"      [{i+1}] Base64: len={len(mask)} chars")
            if len(masks) > 3:
                print(f"      ... and {len(masks)-3} more")
        
    except Exception as e:
        print(f"    ✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # --- Step 4: Run pipeline in QUERY mode ---
    print("\n" + "-" * 60)
    print("[4/5] Running pipeline (mode=query, text_query='car')...")
    print("-" * 60)
    
    try:
        result_query = await pipeline.analyze(
            image=image,
            text_query="car",
            mode="query",
            mask_format="png_base64"
        )
        
        print("\n📊 QUERY MODE RESULTS:")
        print(f"    Mode used: {result_query.get('mode_used')}")
        print(f"    Processing: {result_query.get('meta', {}).get('processing_mode')}")
        print(f"    Objects detected: {result_query.get('meta', {}).get('objects_detected')}")
        print(f"    Masks generated: {result_query.get('masks_generated')}")
        print(f"    Mask format: {result_query.get('mask_format')}")
        
        # Show detections
        detections = result_query.get('detections', [])
        if detections:
            print(f"\n    Detections ({len(detections)}):")
            for i, det in enumerate(detections[:5]):
                print(f"      [{i+1}] {det['label']}: {det['confidence']:.2f}")
        
    except Exception as e:
        print(f"    ✗ Query mode failed: {e}")
    
    # --- Step 5: Run pipeline in SMART QUERY mode ---
    print("\n" + "-" * 60)
    print("[5/5] Running pipeline (mode=smart_query)...")
    print("      User input: 'この画像の中の車や人を見つけて' (Find cars and people)")
    print("-" * 60)
    
    result_smart = None
    try:
        result_smart = await pipeline.analyze(
            image=image,
            user_text="この画像の中の車や人を見つけて",  # Natural language input
            mode="smart_query",
            mask_format="png_base64"
        )
        
        print("\n📊 SMART QUERY MODE RESULTS:")
        print(f"    Mode used: {result_smart.get('mode_used')}")
        print(f"    Processing: {result_smart.get('meta', {}).get('processing_mode')}")
        print(f"    Objects detected: {result_smart.get('meta', {}).get('objects_detected')}")
        print(f"    Masks generated: {result_smart.get('masks_generated')}")
        print(f"    Mask format: {result_smart.get('mask_format')}")
        
        # Show generated queries
        queries_used = result_smart.get('queries_used', [])
        if queries_used:
            print(f"\n    🤖 Gemma3 Generated Queries: {queries_used}")
        
        # Show detections
        detections = result_smart.get('detections', [])
        if detections:
            print(f"\n    Detections ({len(detections)}):")
            for i, det in enumerate(detections[:10]):
                print(f"      [{i+1}] {det['label']}: {det['confidence']:.2f}")
        
    except Exception as e:
        print(f"    ✗ Smart query mode failed: {e}")
        import traceback
        traceback.print_exc()
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"\n💡 Key observations:")
    print(f"   - AUTO mode uses: YOLO → (RF-DETR if dense) → NMS → SAM3")
    print(f"   - QUERY mode uses: SAM3 text-first → fallback to detectors")
    print(f"   - SMART QUERY mode uses: Gemma3 → generates queries → SAM3")
    print(f"   - SAM3 provides pixel-level masks for each detected object")
    print(f"   - MAX_MASK_BOXES={Config.MAX_MASK_BOXES} limits output count")
    
    # Save full results to JSON for inspection
    output_path = Path(__file__).parent / "demo_output.json"
    with open(output_path, 'w') as f:
        # Convert masks to summary for JSON
        result_summary = {
            "auto_mode": {
                **{k: v for k, v in result.items() if k != 'masks'},
                "masks_summary": [
                    f"RLE: size={m.get('size')}" if isinstance(m, dict) else f"base64: {len(m)} chars"
                    for m in result.get('masks', [])[:5]
                ]
            },
            "query_mode": {
                **{k: v for k, v in result_query.items() if k != 'masks'},
                "masks_summary": [
                    f"base64: {len(m)} chars" if isinstance(m, str) else str(type(m))
                    for m in result_query.get('masks', [])[:5]
                ]
            }
        }
        # Add smart query mode if available
        if result_smart:
            result_summary["smart_query_mode"] = {
                **{k: v for k, v in result_smart.items() if k != 'masks'},
                "masks_summary": [
                    f"base64: {len(m)} chars" if isinstance(m, str) else str(type(m))
                    for m in result_smart.get('masks', [])[:5]
                ]
            }
        json.dump(result_summary, f, indent=2, default=str)
    print(f"\n📁 Full results saved to: {output_path}")
    
    # --- Visualize and save detection results ---
    visualize_results(demo_image_path, output_path)


def visualize_results(image_path: Path, json_path: Path):
    """
    Load demo_output.json and draw detection results on demo_input.jpg.
    Saves separate images for auto_mode and query_mode.
    """
    from PIL import ImageDraw, ImageFont
    import random
    
    print("\n" + "=" * 60)
    print("VISUALIZING DETECTION RESULTS")
    print("=" * 60)
    
    # Load the original image and JSON results
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Color palette for different labels
    def get_color(label: str, idx: int = 0) -> tuple:
        """Get a consistent color for each label."""
        colors = {
            'car': (0, 255, 0),        # Green
            'person': (255, 0, 0),      # Red
            'traffic light': (255, 255, 0),  # Yellow
            'truck': (0, 0, 255),       # Blue
            'bus': (255, 165, 0),       # Orange
            'bicycle': (255, 0, 255),   # Magenta
            'motorcycle': (0, 255, 255), # Cyan
        }
        if label in colors:
            return colors[label]
        # Generate a random but consistent color for unknown labels
        random.seed(hash(label))
        return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    
    def draw_detections(image: Image.Image, detections: list, mode_name: str) -> Image.Image:
        """Draw bounding boxes and labels on the image."""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        for i, det in enumerate(detections):
            label = det.get('label', 'unknown')
            confidence = det.get('confidence', 0)
            box = det.get('box', [0, 0, 0, 0])
            
            x1, y1, x2, y2 = [int(coord) for coord in box]
            color = get_color(label, i)
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw label background
            text = f"{label}: {confidence:.2f}"
            bbox = draw.textbbox((x1, y1), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw background rectangle for text
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
            
            # Draw text
            draw.text((x1 + 2, y1 - text_height - 2), text, fill=(255, 255, 255), font=font)
        
        return img_copy
    
    # Process AUTO mode
    if 'auto_mode' in results:
        print("\n[AUTO MODE] Drawing detections...")
        image = Image.open(image_path).convert("RGB")
        auto_detections = results['auto_mode'].get('detections', [])
        auto_image = draw_detections(image, auto_detections, "auto")
        
        auto_output_path = image_path.parent / "demo_output_auto.jpg"
        auto_image.save(auto_output_path, quality=95)
        print(f"    ✓ Saved: {auto_output_path}")
        print(f"    Objects drawn: {len(auto_detections)}")
    
    # Process QUERY mode
    if 'query_mode' in results:
        print("\n[QUERY MODE] Drawing detections...")
        image = Image.open(image_path).convert("RGB")
        query_detections = results['query_mode'].get('detections', [])
        query_image = draw_detections(image, query_detections, "query")
        
        query_output_path = image_path.parent / "demo_output_query.jpg"
        query_image.save(query_output_path, quality=95)
        print(f"    ✓ Saved: {query_output_path}")
        print(f"    Objects drawn: {len(query_detections)}")
    
    # Process SMART QUERY mode
    if 'smart_query_mode' in results:
        print("\n[SMART QUERY MODE] Drawing detections...")
        image = Image.open(image_path).convert("RGB")
        smart_detections = results['smart_query_mode'].get('detections', [])
        smart_image = draw_detections(image, smart_detections, "smart_query")
        
        # Add generated queries info to image
        queries_used = results['smart_query_mode'].get('queries_used', [])
        if queries_used:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(smart_image)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except:
                font = ImageFont.load_default()
            query_text = f"Queries: {', '.join(queries_used)}"
            draw.text((10, 10), query_text, fill=(255, 255, 0), font=font)
        
        smart_output_path = image_path.parent / "demo_output_smart_query.jpg"
        smart_image.save(smart_output_path, quality=95)
        print(f"    ✓ Saved: {smart_output_path}")
        print(f"    Objects drawn: {len(smart_detections)}")
        if queries_used:
            print(f"    Queries used: {queries_used}")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
