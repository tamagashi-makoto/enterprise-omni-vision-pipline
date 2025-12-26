"""
Unit Tests for Omni-Vision Pipeline

Tests pipeline logic with mocked model wrappers.
Does not require GPU or actual models.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image
import numpy as np
import io


# Helper to create minimal test image
def create_test_image(width=100, height=100):
    """Create a simple test PIL Image."""
    return Image.new('RGB', (width, height), color='gray')


def create_minimal_png_bytes():
    """Create valid minimal PNG bytes for API tests."""
    import struct
    import zlib
    
    signature = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack('>IIBBBB', 100, 100, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
    ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
    
    raw_data = b'\x00' + b'\xff\x00\x00' * 100  # filter + RGB per row
    for _ in range(99):  # remaining rows
        raw_data += b'\x00' + b'\xff\x00\x00' * 100
    compressed = zlib.compress(raw_data)
    idat_crc = zlib.crc32(b'IDAT' + compressed) & 0xffffffff
    idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + struct.pack('>I', idat_crc)
    
    iend_crc = zlib.crc32(b'IEND') & 0xffffffff
    iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
    
    return signature + ihdr + idat + iend


# Mock DetectionResult
class MockDetectionResult:
    def __init__(self, label, confidence, box):
        self.label = label
        self.confidence = confidence
        self.box = box
    
    def to_dict(self):
        return {"label": self.label, "confidence": self.confidence, "box": self.box}


# =============================================================================
# BoxSet Helper Tests
# =============================================================================

class TestBoxSetHelpers:
    """Test BoxSet helper functions."""
    
    def test_to_boxset_from_yolo(self):
        """Test conversion from YOLO detections to BoxItems."""
        from src.pipeline import _to_boxset_from_yolo
        
        detections = [
            MockDetectionResult("person", 0.95, [10, 20, 100, 200]),
            MockDetectionResult("car", 0.88, [50, 60, 150, 250]),
        ]
        
        boxset = _to_boxset_from_yolo(detections)
        
        assert len(boxset) == 2
        assert boxset[0].source == "yolo"
        assert boxset[0].label == "person"
        assert boxset[0].score == 0.95
        assert boxset[1].id == "yolo_1"
    
    def test_merge_boxsets_nms(self):
        """Test NMS merging removes duplicates."""
        from src.pipeline import _merge_boxsets_nms, BoxItem
        
        # Two overlapping boxes (IoU > 0.5)
        boxes = [
            BoxItem(id="1", box=[0, 0, 100, 100], score=0.9, source="yolo"),
            BoxItem(id="2", box=[10, 10, 110, 110], score=0.8, source="rfdetr"),  # overlaps with 1
            BoxItem(id="3", box=[200, 200, 300, 300], score=0.7, source="yolo"),  # no overlap
        ]
        
        merged = _merge_boxsets_nms(boxes, iou_threshold=0.5)
        
        # Should keep box 1 (higher score) and box 3 (no overlap)
        assert len(merged) == 2
        assert merged[0].id == "1"  # highest score kept
        assert merged[1].id == "3"
    
    def test_filter_boxes_by_area(self):
        """Test area filtering removes tiny boxes."""
        from src.pipeline import _filter_boxes_by_area, BoxItem
        
        boxes = [
            BoxItem(id="1", box=[0, 0, 100, 100], score=0.9, source="yolo"),  # area=10000
            BoxItem(id="2", box=[0, 0, 5, 5], score=0.8, source="yolo"),      # area=25 (tiny)
        ]
        
        # Image 1000x1000 = 1M pixels, min_ratio=0.0005 = 500px min area
        filtered = _filter_boxes_by_area(boxes, image_size=(1000, 1000), min_area_ratio=0.0005)
        
        assert len(filtered) == 1
        assert filtered[0].id == "1"
    
    def test_select_top_k(self):
        """Test top-K selection."""
        from src.pipeline import _select_top_k, BoxItem
        
        boxes = [
            BoxItem(id="1", box=[0, 0, 10, 10], score=0.5, source="yolo"),
            BoxItem(id="2", box=[0, 0, 10, 10], score=0.9, source="yolo"),
            BoxItem(id="3", box=[0, 0, 10, 10], score=0.7, source="yolo"),
        ]
        
        top2 = _select_top_k(boxes, k=2)
        
        assert len(top2) == 2
        assert top2[0].score == 0.9  # highest first
        assert top2[1].score == 0.7


# =============================================================================
# Pipeline Integration Tests (Mocked)
# =============================================================================

class TestPipelineMocked:
    """Test pipeline logic with mocked model wrappers."""
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create pipeline with mocked wrappers."""
        from src.pipeline import OmniVisionPipeline
        
        pipeline = OmniVisionPipeline()
        pipeline.models_loaded = True
        
        # Mock YOLO
        pipeline.yolo = MagicMock()
        pipeline.yolo.predict = AsyncMock(return_value=[
            MockDetectionResult("person", 0.95, [10, 20, 100, 200]),
            MockDetectionResult("car", 0.88, [150, 160, 250, 350]),
        ])
        
        # Mock RF-DETR
        pipeline.rf_detr = MagicMock()
        pipeline.rf_detr.predict = AsyncMock(return_value=[])
        
        # Mock Florence-2
        pipeline.florence_2 = MagicMock()
        pipeline.florence_2.predict = AsyncMock(return_value=[])
        
        # Mock SAM3
        pipeline.sam_3 = MagicMock()
        pipeline.sam_3.enabled = True
        pipeline.sam_3.predict = AsyncMock(return_value={
            "masks": [np.zeros((100, 100), dtype=bool)],
            "mask_scores": [0.95],
            "mask_boxes": [[10, 20, 100, 200]],
            "metadata": {"source": "box_prompt"}
        })
        
        return pipeline
    
    @pytest.mark.asyncio
    async def test_auto_mode_returns_masks(self, mock_pipeline):
        """Auto mode should return masks from SAM3."""
        image = create_test_image()
        
        result = await mock_pipeline.analyze(image, mode="auto")
        
        assert result["masks_generated"] == 1
        assert result["segmentation_available"] == True
        assert result["mode_used"] == "auto"
        assert len(result["mask_scores"]) == 1
    
    @pytest.mark.asyncio
    async def test_auto_mode_calls_sam3_with_boxes(self, mock_pipeline):
        """Auto mode should call SAM3 with boxes from detectors."""
        image = create_test_image()
        
        await mock_pipeline.analyze(image, mode="auto")
        
        # SAM3 should be called
        mock_pipeline.sam_3.predict.assert_called_once()
        call_kwargs = mock_pipeline.sam_3.predict.call_args[1]
        assert "boxes" in call_kwargs
        assert len(call_kwargs["boxes"]) == 2  # 2 YOLO detections
    
    @pytest.mark.asyncio
    async def test_topk_budget_enforced(self, mock_pipeline):
        """Top-K budget should limit boxes passed to SAM3."""
        from src.config import Config
        
        # Mock YOLO to return 50 boxes
        mock_pipeline.yolo.predict = AsyncMock(return_value=[
            MockDetectionResult(f"obj_{i}", 0.9 - i*0.01, [i*10, i*10, i*10+50, i*10+50])
            for i in range(50)
        ])
        
        image = create_test_image(1000, 1000)
        
        await mock_pipeline.analyze(image, mode="auto")
        
        # SAM3 should receive at most MAX_MASK_BOXES
        call_kwargs = mock_pipeline.sam_3.predict.call_args[1]
        assert len(call_kwargs["boxes"]) <= Config.MAX_MASK_BOXES
    
    @pytest.mark.asyncio
    async def test_query_mode_text_first(self, mock_pipeline):
        """Query mode should try SAM3 text-prompt first."""
        from src.config import Config
        
        # Configure for text-first
        Config.SAM3_TEXT_FIRST = True
        
        image = create_test_image()
        
        await mock_pipeline.analyze(image, text_query="person", mode="query")
        
        # SAM3 should be called with text_prompt
        mock_pipeline.sam_3.predict.assert_called()
        call_kwargs = mock_pipeline.sam_3.predict.call_args[1]
        assert call_kwargs.get("text_prompt") == "person"
    
    @pytest.mark.asyncio
    async def test_query_mode_fallback_if_empty(self, mock_pipeline):
        """Query mode should fall back to detectors if SAM3 text returns empty."""
        from src.config import Config
        Config.SAM3_TEXT_FIRST = True
        Config.FALLBACK_IF_NO_MASK = True
        
        # First SAM3 call returns empty, second returns masks
        call_count = [0]
        async def mock_sam3_predict(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:  # First call (text-prompted)
                return {"masks": [], "mask_scores": [], "mask_boxes": [], "metadata": {}}
            else:  # Second call (box-prompted)
                return {"masks": [np.zeros((100, 100))], "mask_scores": [0.9], "mask_boxes": [[10, 20, 100, 200]], "metadata": {}}
        
        mock_pipeline.sam_3.predict = mock_sam3_predict
        
        image = create_test_image()
        result = await mock_pipeline.analyze(image, text_query="person", mode="query")
        
        # Should have fallen back to detector boxes
        assert call_count[0] == 2  # SAM3 called twice
    
    @pytest.mark.asyncio
    async def test_density_triggers_rfdetr(self, mock_pipeline):
        """High density should trigger RF-DETR."""
        from src.config import Config
        
        # Mock YOLO to return > DENSITY_THRESHOLD boxes
        mock_pipeline.yolo.predict = AsyncMock(return_value=[
            MockDetectionResult(f"obj_{i}", 0.9, [i*10, i*10, i*10+20, i*10+20])
            for i in range(Config.DENSITY_THRESHOLD + 5)
        ])
        
        image = create_test_image(1000, 1000)
        
        await mock_pipeline.analyze(image, mode="auto")
        
        # RF-DETR should be called
        mock_pipeline.rf_detr.predict.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_response_includes_mask_fields(self, mock_pipeline):
        """Response should include mask_scores and mask_boxes."""
        image = create_test_image()
        
        result = await mock_pipeline.analyze(image, mode="auto")
        
        assert "mask_scores" in result
        assert "mask_boxes" in result
        assert "mode_used" in result
    
    @pytest.mark.asyncio
    async def test_sam3_disabled_degrades_gracefully(self, mock_pipeline):
        """Pipeline should work even if SAM3 is disabled."""
        mock_pipeline.sam_3.enabled = False
        
        image = create_test_image()
        result = await mock_pipeline.analyze(image, mode="auto")
        
        # Should still return detections, just no masks
        assert result["masks_generated"] == 0
        assert result["segmentation_available"] == False
        assert len(result["detections"]) == 2  # YOLO detections


# =============================================================================
# API Tests
# =============================================================================

@pytest.mark.asyncio
async def test_health_check():
    """Test health check endpoint."""
    from httpx import AsyncClient, ASGITransport
    from src.main import app
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "cuda_available" in data
    assert "sam3_enabled" in data


@pytest.mark.asyncio
async def test_analyze_with_mode_parameter():
    """Test /analyze accepts mode parameter."""
    from httpx import AsyncClient, ASGITransport
    from src.main import app
    
    file = io.BytesIO(create_minimal_png_bytes())
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/analyze",
            files={"file": ("test.png", file, "image/png")},
            data={"mode": "auto"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "mode_used" in data


@pytest.mark.asyncio
async def test_invalid_mode_rejected():
    """Test /analyze rejects invalid mode."""
    from httpx import AsyncClient, ASGITransport
    from src.main import app
    
    file = io.BytesIO(create_minimal_png_bytes())
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/analyze",
            files={"file": ("test.png", file, "image/png")},
            data={"mode": "invalid_mode"}
        )
    
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_invalid_file_type():
    """Test rejection of non-image file types."""
    from httpx import AsyncClient, ASGITransport
    from src.main import app
    
    file = io.BytesIO(b"text content")
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/analyze", 
            files={"file": ("test.txt", file, "text/plain")}
        )
    
    assert response.status_code == 400


# =============================================================================
# Mask Serialization Tests
# =============================================================================

class TestMaskSerialization:
    """Test mask serialization functions."""
    
    def test_encode_rle_simple(self):
        """Test RLE encoding of a simple mask."""
        from src.pipeline import encode_rle
        
        # Simple 4x4 mask with center filled
        mask = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ], dtype=np.uint8)
        
        rle = encode_rle(mask)
        
        assert rle["size"] == [4, 4]
        assert rle["order"] == "F"
        assert isinstance(rle["counts"], list)
        assert len(rle["counts"]) > 0
    
    def test_mask_to_png_base64(self):
        """Test PNG base64 encoding of mask."""
        from src.pipeline import mask_to_png_base64
        import base64
        
        mask = np.ones((50, 50), dtype=np.uint8)
        
        b64_str = mask_to_png_base64(mask)
        
        # Should be valid base64
        decoded = base64.b64decode(b64_str)
        assert decoded[:8] == b'\x89PNG\r\n\x1a\n'  # PNG signature
    
    def test_serialize_masks_rle(self):
        """Test serialization with RLE format."""
        from src.pipeline import serialize_masks
        
        masks = [np.zeros((10, 10), dtype=np.uint8)]
        
        serialized = serialize_masks(masks, "rle")
        
        assert len(serialized) == 1
        assert "size" in serialized[0]
        assert "counts" in serialized[0]
    
    def test_serialize_masks_png_base64(self):
        """Test serialization with PNG base64 format."""
        from src.pipeline import serialize_masks
        
        masks = [np.ones((10, 10), dtype=np.uint8)]
        
        serialized = serialize_masks(masks, "png_base64")
        
        assert len(serialized) == 1
        assert isinstance(serialized[0], str)
    
    def test_serialize_masks_none_format(self):
        """Test serialization with 'none' format returns empty."""
        from src.pipeline import serialize_masks
        
        masks = [np.ones((10, 10), dtype=np.uint8)]
        
        serialized = serialize_masks(masks, "none")
        
        assert serialized == []


class TestPipelineMasksInResponse:
    """Test that masks are properly included in pipeline response."""
    
    @pytest.fixture
    def mock_pipeline_with_masks(self):
        """Create pipeline that returns actual mask data."""
        from src.pipeline import OmniVisionPipeline
        
        pipeline = OmniVisionPipeline()
        pipeline.models_loaded = True
        
        # Mock YOLO
        pipeline.yolo = MagicMock()
        pipeline.yolo.predict = AsyncMock(return_value=[
            MockDetectionResult("person", 0.95, [10, 20, 100, 200]),
        ])
        
        # Mock RF-DETR
        pipeline.rf_detr = MagicMock()
        pipeline.rf_detr.predict = AsyncMock(return_value=[])
        
        # Mock Florence-2
        pipeline.florence_2 = MagicMock()
        pipeline.florence_2.predict = AsyncMock(return_value=[])
        
        # Mock SAM3 with actual mask data
        pipeline.sam_3 = MagicMock()
        pipeline.sam_3.enabled = True
        pipeline.sam_3.predict = AsyncMock(return_value={
            "masks": [np.ones((50, 50), dtype=np.uint8)],
            "mask_scores": [0.95],
            "mask_boxes": [[10, 20, 100, 200]],
            "metadata": {"source": "box_prompt"}
        })
        
        return pipeline
    
    @pytest.mark.asyncio
    async def test_response_includes_masks_rle(self, mock_pipeline_with_masks):
        """Response should include RLE-encoded masks by default."""
        image = create_test_image()
        
        result = await mock_pipeline_with_masks.analyze(image, mode="auto", mask_format="rle")
        
        assert "masks" in result
        assert len(result["masks"]) == 1
        assert "size" in result["masks"][0]
        assert "counts" in result["masks"][0]
        assert result["mask_format"] == "rle"
    
    @pytest.mark.asyncio
    async def test_response_includes_masks_png(self, mock_pipeline_with_masks):
        """Response should include PNG base64 masks when requested."""
        image = create_test_image()
        
        result = await mock_pipeline_with_masks.analyze(image, mode="auto", mask_format="png_base64")
        
        assert "masks" in result
        assert len(result["masks"]) == 1
        assert isinstance(result["masks"][0], str)
        assert result["mask_format"] == "png_base64"
    
    @pytest.mark.asyncio
    async def test_response_masks_empty_when_none(self, mock_pipeline_with_masks):
        """Response should have empty masks when format is 'none'."""
        image = create_test_image()
        
        result = await mock_pipeline_with_masks.analyze(image, mode="auto", mask_format="none")
        
        assert "masks" in result
        assert result["masks"] == []
        assert result["mask_format"] == "none"


# =============================================================================
# Integration Tests (GPU required, gated by env var)
# =============================================================================

import os

@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_real_pipeline():
    """Integration test with real models (requires GPU)."""
    if not os.environ.get("RUN_INTEGRATION_TESTS"):
        pytest.skip("Integration tests disabled (set RUN_INTEGRATION_TESTS=1 to enable)")
    
    from httpx import AsyncClient, ASGITransport
    from src.main import app
    
    file = io.BytesIO(create_minimal_png_bytes())
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", timeout=120.0) as ac:
        response = await ac.post(
            "/analyze",
            files={"file": ("test.png", file, "image/png")},
            data={"mode": "auto", "mask_format": "rle"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "masks" in data
    assert "mask_scores" in data


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_query_mode_with_fallback():
    """Integration test for query mode with fallback (requires GPU)."""
    if not os.environ.get("RUN_INTEGRATION_TESTS"):
        pytest.skip("Integration tests disabled (set RUN_INTEGRATION_TESTS=1 to enable)")
    
    from httpx import AsyncClient, ASGITransport
    from src.main import app
    
    file = io.BytesIO(create_minimal_png_bytes())
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", timeout=120.0) as ac:
        response = await ac.post(
            "/analyze",
            files={"file": ("test.png", file, "image/png")},
            data={"mode": "query", "text_query": "object"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["mode_used"] == "query"
