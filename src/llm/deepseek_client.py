# ===============================
# src/llm/deepseek_client.py
# ===============================

import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gpt-oss:120b-cloud"
# gpt-oss:120b-cloud
# gemma4:31b-cloud

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