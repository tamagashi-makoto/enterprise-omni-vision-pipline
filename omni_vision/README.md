# Omni-Vision-Analytics

**Omni-Vision-Analytics** is an intelligent orchestration pipeline that dynamically selects the best computer vision model based on scene complexity and user intent. Designed as a modular, high-performance microservice.

## 🧠 System Architecture

The pipeline orchestrates the following models (mocked for this demonstration):
1.  **Stage 1: Screening (YOLOv12)** - Fast initial detection.
2.  **Stage 2: Density Check (RF-DETR)** - Activated if object count > 15.
3.  **Stage 3: Intent Check (DINO-X)** - Activated if a specific `text_query` is provided.
4.  **Stage 4: Segmentation (SAM 3)** - Generates masks for finalized detections.

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **API:** FastAPI (Async)
- **Containerization:** Docker & Docker Compose
- **Validation:** Pydantic

## 🚀 Getting Started

### Prerequisites
- Docker & Docker Compose
- Python 3.10+ (for local dev)

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
python -m uvicorn src.main:app --reload
```

## 🔌 API Endpoints

### `POST /analyze`
Analyze an image.

**Parameters:**
- `file`: (Required) Image file.
- `text_query`: (Optional) String describing object to find (e.g., "red backpack").

**Response:**
```json
{
  "meta": {
    "processing_mode": "string",
    "objects_detected": "int"
  },
  "detections": [
    {
      "label": "string",
      "confidence": "float",
      "box": [x1, y1, x2, y2]
    }
  ],
  "segmentation_available": "bool"
}
```

### `GET /health`
Check system status.

## 📂 Project Structure
```text
omni_vision/
├── src/
│   ├── main.py            # FastAPI Entry Point
│   ├── pipeline.py        # Orchestration Logic
│   ├── model_wrappers.py  # Model Interfaces & Mocks
│   └── config.py          # Configuration
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
