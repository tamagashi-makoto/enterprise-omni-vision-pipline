import pytest
from httpx import AsyncClient, ASGITransport
from src.main import app
from src.config import ModelType
import io

@pytest.mark.asyncio
async def test_health_check():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_analyze_default_yolo():
    # Create a dummy image file
    file = io.BytesIO(b"fakeimagebytes")
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/analyze", 
            files={"file": ("test.jpg", file, "image/jpeg")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "YOLO" in data["meta"]["processing_mode"]
    assert data["segmentation_available"] is True
    assert len(data["detections"]) > 0

@pytest.mark.asyncio
async def test_analyze_with_text_query():
    file = io.BytesIO(b"fakeimagebytes")
    query = "lost keys"
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/analyze", 
            files={"file": ("test.jpg", file, "image/jpeg")},
            data={"text_query": query}
        )
    
    assert response.status_code == 200
    data = response.json()
    # Expect DINO-X to be active
    assert ModelType.DINO_X in data["meta"]["processing_mode"]
    assert query in data["meta"]["processing_mode"]
    
@pytest.mark.asyncio
async def test_invalid_file_type():
    file = io.BytesIO(b"text content")
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/analyze", 
            files={"file": ("test.txt", file, "text/plain")}
        )
    
    assert response.status_code == 400
