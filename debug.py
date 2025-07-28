import httpx

API_KEY = "GeGcfonMU5A7huGpAy1EXw4QJc7T5mTi"
API_URL = "https://api.mistral.ai/v1/chat/completions"


def test_mistral_api():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "mistral-small-latest",
        "messages": [
            {"role": "user", "content": "Hello, can you confirm if my API key works?"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }

    try:
        response = httpx.post(API_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        print("✅ API Response:")
        print(data)
    except httpx.HTTPStatusError as e:
        print(f"❌ HTTP error occurred: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")


if __name__ == "__main__":
    test_mistral_api()
