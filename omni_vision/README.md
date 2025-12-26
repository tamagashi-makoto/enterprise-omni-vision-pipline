# Omni-Vision-Analytics

**Omni-Vision-Analytics** is an intelligent orchestration pipeline that dynamically selects the best computer vision model based on scene complexity and user intent. SAM3 serves as the final source of truth for pixel-level masks.

## 🎯 Use Cases

### 1. E-commerce Cutout (Background Removal)
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@product.jpg"
```
Uses: YOLO → NMS merge → SAM3 (box prompts)

### 2. Privacy Masking (Faces, License Plates)
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@photo.jpg" \
  -F "text_query=face, license plate" \
  -F "mode=query"
```
Uses: SAM3 (text prompt) → fallback to YOLO/RF-DETR if needed

### 3. Manufacturing Visual Inspection
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@pcb.jpg" \
  -F "text_query=scratch, crack, defect" \
  -F "mode=query"
```
Uses: SAM3 (text prompt) → Florence-2 rerank (optional) → SAM3 (box prompts)

## 🧠 System Architecture

```
┌─────────────┐     ┌─────────────┐
│  YOLOv12    │────>│  BoxSet     │
│  (Fast)     │     │  Merge      │
└─────────────┘     │  (NMS)      │     ┌─────────────┐
                    │             │────>│  SAM3       │
┌─────────────┐     │  Filter     │     │  (Masks)    │
│  RF-DETR    │────>│  (Area)     │     └─────────────┘
│  (Dense)    │     │             │
└─────────────┘     │  Top-K      │
                    └─────────────┘
┌─────────────┐
│ Florence-2  │ (Optional rerank in query mode)
└─────────────┘
```

### Inference Modes

| Mode | Flow | Use Case |
|------|------|----------|
| `auto` (default) | YOLO → (RF-DETR if dense) → merge → Top-K → SAM3 (box prompt) | E-commerce, baseline privacy |
| `query` | SAM3 (text-first) → fallback to detectors → SAM3 (box prompt) | Open-vocabulary, defect discovery |

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **API:** FastAPI (Async)
- **Models:** YOLOv12, RF-DETR, Florence-2, SAM3
- **Containerization:** Docker & Docker Compose

## 🚀 Getting Started

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with CUDA (recommended for SAM3)
- HuggingFace token (for SAM3): `export HF_TOKEN=your_token`

### Running with Docker (Recommended)
```bash
# Build and Run
docker-compose up --build
```
The API will be available at `http://localhost:8000`.

### Running Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run Server
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🔌 API Endpoints

### `GET /health`
Check system status including SAM3 availability.

```json
{
  "status": "healthy",
  "models_loaded": true,
  "cuda_available": true,
  "sam3_enabled": true
}
```

### `POST /analyze`
Analyze an image using the intelligent model pipeline.

**Parameters:**
- `file`: (Required) Image file
- `text_query`: (Optional) Text query for open-vocabulary detection
- `mode`: (Optional) `"auto"` or `"query"` (default: `"auto"`)

**Response:**
```json
{
  "meta": {
    "processing_mode": "YOLO(5) → RF-DETR(3)",
    "objects_detected": 5
  },
  "detections": [
    {"label": "person", "confidence": 0.95, "box": [10, 20, 100, 200], "has_mask": true}
  ],
  "segmentation_available": true,
  "masks_generated": 5,
  "mask_scores": [0.95, 0.92, ...],
  "mask_boxes": [[10, 20, 100, 200], ...],
  "mode_used": "auto"
}
```

## 🧪 Running Tests

### Unit Tests (Mocked, CI-friendly)
```bash
cd omni_vision
pytest tests/test_pipeline_unit.py -v
```

### API Tests
```bash
cd omni_vision
pytest tests/test_api.py -v
```

### Integration Tests (GPU required)
```bash
export RUN_INTEGRATION_TESTS=1
pytest tests/ -v -m integration
```

### Stress Test (Server must be running)
```bash
python tests/stress_test.py
```

## 📂 Project Structure
```text
omni_vision/
├── src/
│   ├── main.py            # FastAPI Entry Point
│   ├── pipeline.py        # BoxSet Orchestration Logic
│   ├── model_wrappers.py  # Model Interfaces (YOLO, RF-DETR, Florence-2, SAM3)
│   └── config.py          # Configuration
├── tests/
│   ├── test_pipeline_unit.py  # Unit tests with mocks
│   ├── test_api.py            # API endpoint tests
│   └── stress_test.py         # Load testing
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## ⚙️ Configuration

Key settings in `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_MASK_BOXES` | 20 | Top-K budget for SAM3 |
| `NMS_IOU_THRESHOLD` | 0.5 | NMS merge threshold |
| `DENSITY_THRESHOLD` | 15 | Trigger RF-DETR if YOLO finds more |
| `SAM3_TEXT_FIRST` | True | Try text-prompt before detectors in query mode |
| `ENABLE_FLORENCE_RERANK` | False | Enable Florence-2 reranking |
