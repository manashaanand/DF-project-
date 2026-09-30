import httpx; print(httpx.post('http://127.0.0.1:8000/multimedia/analyze', files={'file': open('test_appended.png', 'rb')}, timeout=60.0).text)
