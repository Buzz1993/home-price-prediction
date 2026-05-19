# # ===============================
# # src/llm/deepseek_memory.py
# # ===============================

from src.llm.deepseek_client import ask_deepseek

SYSTEM_PROMPT = """
You are a smart real estate assistant with memory.

Use user memory to personalize responses.
- Always use user's name if available
- Do NOT hallucinate
- Answer directly
- Use property data only if relevant

At the end suggest 3 follow-up questions.
"""


def extract_memory(user_msg):

    prompt = f"""
    Extract user memory from message.

    Message: {user_msg}

    Return short memory sentence or NONE.
    """

    res = ask_deepseek(prompt).strip()

    if res.lower() == "none":
        return None

    return res


def build_prompt(user_msg, history, memory, context):

    history_text = ""
    for role, msg in history:
        if role == "You":
            history_text += f"User: {msg}\n"
        else:
            history_text += f"Assistant: {msg}\n"

    memory_text = "\n".join(memory) if memory else "None"

    prompt = f"""
    {SYSTEM_PROMPT}

    USER MEMORY:
    {memory_text}

    CHAT HISTORY:
    {history_text}

    PROPERTY DATA:
    {context}

    USER: {user_msg}
    ASSISTANT:
    """

    return prompt