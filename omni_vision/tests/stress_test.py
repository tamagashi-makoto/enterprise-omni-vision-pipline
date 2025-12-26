"""
Stress Test for Omni-Vision API

Tests concurrency and edge cases with valid image data.
Run against a live server: python tests/stress_test.py
"""
import asyncio
import httpx
import time
import io
import random
import string
import struct
import zlib

BASE_URL = "http://localhost:8001"


def create_minimal_png() -> bytes:
    """Create a valid minimal PNG image for testing."""
    signature = b'\x89PNG\r\n\x1a\n'
    
    # IHDR chunk (100x100, 8-bit RGB)
    ihdr_data = struct.pack('>IIBBBBB', 100, 100, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
    ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
    
    # IDAT chunk (compressed image data)
    raw_data = b''
    for _ in range(100):
        raw_data += b'\x00' + bytes([random.randint(0, 255) for _ in range(300)])
    compressed = zlib.compress(raw_data)
    idat_crc = zlib.crc32(b'IDAT' + compressed) & 0xffffffff
    idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + struct.pack('>I', idat_crc)
    
    # IEND chunk
    iend_crc = zlib.crc32(b'IEND') & 0xffffffff
    iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
    
    return signature + ihdr + idat + iend


async def send_valid_request(client: httpx.AsyncClient, i: int):
    """Sends a valid image analysis request."""
    file_content = create_minimal_png()
    files = {"file": (f"test_{i}.png", file_content, "image/png")}
    
    start = time.time()
    try:
        response = await client.post("/analyze", files=files)
        response.raise_for_status()
        duration = time.time() - start
        return "OK", duration
    except Exception as e:
        return f"ERR: {e}", 0


async def send_request_with_mode(client: httpx.AsyncClient, mode: str):
    """Sends a request with specific mode."""
    file_content = create_minimal_png()
    files = {"file": ("test.png", file_content, "image/png")}
    data = {"mode": mode}
    
    try:
        response = await client.post("/analyze", files=files, data=data)
        if response.status_code == 200:
            body = response.json()
            return f"OK (mode_used: {body.get('mode_used', 'unknown')})", 0
        elif response.status_code == 400 and mode not in ("auto", "query"):
            return "OK (400 for invalid mode)", 0
        return f"FAIL: Status {response.status_code}", 0
    except Exception as e:
        return f"ERR: {e}", 0


async def send_malformed_request(client: httpx.AsyncClient):
    """Sends a request with invalid content type."""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    try:
        response = await client.post("/analyze", files=files)
        if response.status_code == 400:
            return "OK (400 caught)", 0
        return f"FAIL: Status {response.status_code}", 0
    except Exception as e:
        return f"ERR: {e}", 0


async def stress_test():
    print(f"Starting Stress Test against {BASE_URL}...")
    
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        # Pre-check
        try:
            r = await client.get("/health")
            r.raise_for_status()
            health = r.json()
            print("Health Check: PASSED")
            print(f"  - CUDA: {health.get('cuda_available', 'unknown')}")
            print(f"  - SAM3: {health.get('sam3_enabled', 'unknown')}")
        except Exception as e:
            print(f"Health Check: FAILED ({e}). Is server running?")
            return

        # 1. Concurrency Test
        print("\n--- Phase 1: Concurrency (50 requests) ---")
        tasks = [send_valid_request(client, i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        
        success_count = sum(1 for res, _ in results if res == "OK")
        durations = [t for _, t in results if t > 0]
        avg_lat = sum(durations) / len(durations) if durations else 0
        print(f"Success: {success_count}/50")
        print(f"Avg Latency: {avg_lat:.3f}s")
        assert success_count >= 45, f"Concurrency test failed! Only {success_count}/50 succeeded"

        # 2. Edge Cases
        print("\n--- Phase 2: Edge Cases ---")
        
        # Malformed
        res, _ = await send_malformed_request(client)
        print(f"Malformed Content-Type: {res}")
        assert "OK" in res

        # Invalid mode
        res, _ = await send_request_with_mode(client, "invalid_mode")
        print(f"Invalid mode: {res}")
        assert "OK" in res or "400" in res

        # Valid modes
        for mode in ["auto", "query"]:
            res, _ = await send_request_with_mode(client, mode)
            print(f"Mode '{mode}': {res}")
            assert "OK" in res

        # 3. Response Format
        print("\n--- Phase 3: Response Format ---")
        file_content = create_minimal_png()
        resp = await client.post(
            "/analyze",
            files={"file": ("test.png", file_content, "image/png")}
        )
        body = resp.json()
        
        required_fields = ["meta", "detections", "segmentation_available", "masks_generated", "mask_scores", "mask_boxes", "mode_used"]
        for field in required_fields:
            if field in body:
                print(f"  ✓ {field}: present")
            else:
                print(f"  ✗ {field}: MISSING")
        
        missing = [f for f in required_fields if f not in body]
        assert len(missing) == 0, f"Missing fields: {missing}"

    print("\n[SUCCESS] All stress tests passed.")


if __name__ == "__main__":
    asyncio.run(stress_test())
