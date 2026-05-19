# ===============================
# memory_node.py
# ===============================

from src.llm.memory_store import SQLiteMemoryStore
from src.llm.deepseek_memory import extract_memory

memory_store = SQLiteMemoryStore()

USER_ID = "default_user" # Static user ID - "default_user" used for this project because
                         # multi-user support is not implemented yet.
                         # Every time memory_node runs, it uses the same
                         # user_id to store and retrieve memories.


def memory_node(state):
    print("✅ memory_node executed")

    user_msg = state["user_message"]

    # -----------------------------
    # EXTRACT NEW MEMORY
    # -----------------------------
    memory = extract_memory(user_msg)

    if memory:
        memory_store.add_memory(USER_ID, memory)

    # -----------------------------
    # LOAD ALL MEMORIES
    # -----------------------------
    memories = memory_store.get_memories(USER_ID)

    state["memory"] = memories

    return state