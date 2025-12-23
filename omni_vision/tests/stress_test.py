import asyncio
import httpx
import time
import io
import random
import string

BASE_URL = "http://localhost:8001"

async def send_valid_request(client, i):
    """Sends a valid image analysis request."""
    # Create a small valid mock image (random bytes, but we rely on mock to just accept bytes or we make a real one)
    # Since our mocked pipeline just reads bytes, random content is fine as long as main.py check passes.
    # main.py checks `file.content_type.startswith("image/")`.
    # But Pillow or models might choke if we actually processed it.
    # The current mock implementation just sleeps, so random bytes are safe.
    
    file_content = b"fake_image_bytes_" + str(i).encode()
    files = {"file": ("test.jpg", file_content, "image/jpeg")}
    
    start = time.time()
    try:
        response = await client.post("/analyze", files=files)
        response.raise_for_status()
        duration = time.time() - start
        return "OK", duration
    except Exception as e:
        return f"ERR: {e}", 0

async def send_malformed_request(client):
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
    
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
        # Pre-check
        try:
            r = await client.get("/health")
            r.raise_for_status()
            print("Health Check: PASSED")
        except:
            print("Health Check: FAILED. Is server running?")
            return

        # 1. Concurrency Test
        print("\n--- Phase 1: Concurrency (50 requests) ---")
        tasks = [send_valid_request(client, i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        
        success_count = sum(1 for res, _ in results if res == "OK")
        avg_lat = sum(t for _, t in results) / len(results) if results else 0
        print(f"Success: {success_count}/50")
        print(f"Avg Latency (incl. mock sleep): {avg_lat:.3f}s")
        assert success_count == 50, "Concurrency test failed!"

        # 2. Edge Cases
        print("\n--- Phase 2: Edge Cases ---")
        
        # Malformed
        res, _ = await send_malformed_request(client)
        print(f"Malformed Content-Type: {res}")
        assert "OK" in res

        # Huge Text Query
        huge_query = "".join(random.choices(string.ascii_letters, k=5000))
        files = {"file": ("test.jpg", b"img", "image/jpeg")}
        data = {"text_query": huge_query}
        resp = await client.post("/analyze", files=files, data=data)
        print(f"Huge Query (5000 chars): Status {resp.status_code}")
        assert resp.status_code == 200

    print("\n[SUCCESS] All stress tests passed.")

if __name__ == "__main__":
    asyncio.run(stress_test())
