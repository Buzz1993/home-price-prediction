# ===============================
# src/llm/deepseek_client.py
# ===============================

import os

import requests
import json

from dotenv import load_dotenv

load_dotenv()

# Deployment configuration (Railway/production parity):
#   OLLAMA_URL     — generate endpoint. Defaults to the local Ollama daemon,
#                    so local behaviour is unchanged. In production point it
#                    at the hosted Ollama Cloud API
#                    (https://ollama.com/api/generate).
#   OLLAMA_MODEL   — model name. Defaults to the existing local alias.
#                    On Ollama Cloud the same model is named "gpt-oss:120b".
#   OLLAMA_API_KEY — bearer token for the hosted API. Unset locally.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
# gpt-oss:120b-cloud
# gemma4:31b-cloud

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()


def _request_headers() -> dict:
    """Authorization header for the hosted Ollama API (empty locally)."""
    if OLLAMA_API_KEY:
        return {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    return {}

def ask_deepseek(prompt):

    """
    Send prompt to Ollama LLM and return full response.

    Args:
        prompt (str): Input prompt text.

    Returns:
        str: Generated response or error message.
    """
     
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,             #Wait for COMPLETE response first
        "options": {
            "temperature": 0,        #Controls randomness, 0 = deterministic, high value = more creative/random
            "top_p": 1,
            "repeat_penalty": 1
        }
    }

    try:
        response = requests.post( # Make HTTP POST request to Ollama API, with the prompt and options
            OLLAMA_URL,
            json=payload,
            headers=_request_headers(),
            timeout=180
        )

    # if any exception occurs during try block (like connection error, timeout, etc), catch it and print details for debugging, then return an error message.
    except Exception as e:       
        print("\n========== REQUEST EXCEPTION ==========")
        print(type(e)) # This will print the type of exception that occurred, such as ConnectionError, Timeout, etc.
        print(str(e)) # This will print the error message associated with the exception, providing more details about what went wrong.
        print("=======================================\n")
        return f"REQUEST EXCEPTION: {str(e)}"

    # Check if the response status code is not 200 (which means the request was not successful). If it's an error, print the status code and raw response for debugging, 
    # then return an error message.
    if response.status_code != 200:
        print("\n========== OLLAMA ERROR ==========")
        print("STATUS CODE:", response.status_code)
        print("RAW RESPONSE:")
        print(response.text)
        print("==================================\n")
        return (
            f"OLLAMA ERROR ({response.status_code})\n"
            f"{response.text}"
        )

    return response.json().get("response", "")


def ask_deepseek_stream(prompt):

    """
    Send prompt to Ollama LLM and stream response in small chunks
    instead of waiting for the full response.

    Args:
        prompt (str): Input prompt text.

    Yields:
        str: Partial response chunks or error message.
    """

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "repeat_penalty": 1
        }
    }

    try:

        with requests.post(
            OLLAMA_URL,
            json=payload,
            headers=_request_headers(),
            stream=True,
            timeout=180
        ) as response:

            if response.status_code != 200:

                print("\n========== STREAM OLLAMA ERROR ==========")
                print("STATUS CODE:", response.status_code)
                print("RAW RESPONSE:")
                print(response.text)
                print("=========================================\n")

                yield (
                    f"OLLAMA STREAM ERROR ({response.status_code})\n"
                    f"{response.text}"
                )

                return

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        yield data.get("response", "")
                    except:
                        continue

    except Exception as e:

        print("\n========== STREAM EXCEPTION ==========")
        print(type(e))
        print(str(e))
        print("======================================\n")

        yield f"STREAM EXCEPTION: {str(e)}"