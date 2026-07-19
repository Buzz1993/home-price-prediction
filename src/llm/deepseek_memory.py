# ===============================
# src/llm/deepseek_memory.py
# ===============================

from src.llm.provider_router import ask_llm

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
    """
    Sends the user message to the DeepSeek model
    to extract useful user memory from it.
    """

    prompt = f"""
    Extract ONLY long-term useful user preferences
    or requirements from the message.

    Examples of GOOD memory:
    - "I want 2 BHK in Thane"
    - "My budget is 2 Cr"
    - "I prefer investment properties"
    - "I want low-risk properties"

    Examples of BAD memory:
    - "show in table"
    - "only give id and price"
    - "predict again"
    - temporary formatting instructions
    - temporary tasks/questions

    If no useful long-term memory exists, return ONLY:
    NONE

    Message:
    {user_msg}
    """

    res = ask_llm(prompt).strip() # Get the response from ask_llm and remove any leading/trailing whitespace

    if res.lower() == "none":
        return None

    return res


def build_prompt(user_msg, history, memory, context):
    """
    Builds the final LLM prompt using memory,
    chat history, property context, and user query.
    """

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