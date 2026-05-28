import streamlit as st
import requests
import json

st.title("Ollama Stream Test")

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "ministral-3:3b-cloud"
#ministral-3:3b-cloud

prompt = st.text_input("Prompt", "Hello Mumbai")

if st.button("Run"):

    placeholder = st.empty()

    full_response = ""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True
    }

    try:

        with requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=120
        ) as response:

            st.write("Status:", response.status_code)

            for line in response.iter_lines():

                if not line:
                    continue

                try:

                    data = json.loads(
                        line.decode("utf-8")
                    )

                    chunk = data.get("response", "")

                    full_response += chunk

                    placeholder.markdown(full_response + "▌")

                except Exception as e:

                    st.error(f"Chunk Error: {e}")

            placeholder.markdown(full_response)

    except Exception as e:

        st.error(f"Main Error: {e}")