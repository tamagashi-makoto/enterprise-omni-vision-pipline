#!/usr/bin/env python3
"""
Standalone script to visualize detection results.

Reads demo_output.json and draws bounding boxes on demo_input.jpg,
saving separate images for auto_mode and query_mode.

Usage: python visualize_results.py
"""
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


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


def visualize_results(image_path: Path, json_path: Path):
    """
    Load demo_output.json and draw detection results on demo_input.jpg.
    Saves separate images for auto_mode and query_mode.
    """
    print("=" * 60)
    print("VISUALIZING DETECTION RESULTS")
    print("=" * 60)
    
    # Load the JSON results
    with open(json_path, 'r') as f:
        results = json.load(f)
    
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
        
        # Print detection summary
        labels = {}
        for det in auto_detections:
            label = det.get('label', 'unknown')
            labels[label] = labels.get(label, 0) + 1
        print(f"    Detection breakdown: {labels}")
    
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
        
        # Print detection summary
        labels = {}
        for det in query_detections:
            label = det.get('label', 'unknown')
            labels[label] = labels.get(label, 0) + 1
        print(f"    Detection breakdown: {labels}")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    image_path = script_dir / "demo_input.jpg"
    json_path = script_dir / "demo_output.json"
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return 1
    
    if not json_path.exists():
        print(f"Error: JSON not found: {json_path}")
        return 1
    
    visualize_results(image_path, json_path)
    return 0


if __name__ == "__main__":
    exit(main())
