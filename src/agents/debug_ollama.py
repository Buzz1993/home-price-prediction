import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "ministral-3:3b-cloud"


def test_normal():

    print("\n========== NORMAL TEST ==========\n")

    payload = {
        "model": MODEL_NAME,
        "prompt": "Hello",
        "stream": False
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=60
        )

        print("STATUS CODE:", response.status_code)

        print("\nRAW RESPONSE:")
        print(response.text)

        print("\nJSON RESPONSE:")
        print(response.json())

    except Exception as e:
        print("\nERROR:")
        print(e)


def test_stream():

    print("\n========== STREAM TEST ==========\n")

    payload = {
        "model": MODEL_NAME,
        "prompt": "Tell me about Mumbai",
        "stream": True
    }

    try:

        with requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=60
        ) as response:

            print("STATUS CODE:", response.status_code)

            for line in response.iter_lines():

                if not line:
                    continue

                try:

                    decoded = line.decode("utf-8")

                    print("\nRAW CHUNK:")
                    print(decoded)

                    data = json.loads(decoded)

                    print("\nPARSED:")
                    print(data.get("response", ""))

                except Exception as e:
                    print("\nCHUNK ERROR:")
                    print(e)

    except Exception as e:
        print("\nSTREAM ERROR:")
        print(e)


if __name__ == "__main__":

    test_normal()

    test_stream()