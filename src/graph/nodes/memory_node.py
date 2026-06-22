# # ===============================
# # memory_node.py
# # ===============================

# from src.llm.memory_store import SQLiteMemoryStore
# from src.llm.deepseek_memory import extract_memory

# memory_store = SQLiteMemoryStore()

# USER_ID = "default_user" # Static user ID - "default_user" used for this project because
#                          # multi-user support is not implemented yet.
#                          # Every time memory_node runs, it uses the same
#                          # user_id to store and retrieve memories.


# def memory_node(state):
#     print("✅ memory_node executed")

#     user_msg = state["user_message"]

#     # -----------------------------
#     # EXTRACT NEW MEMORY
#     # -----------------------------
#     memory = extract_memory(user_msg)

#     if memory:
#         memory_store.add_memory(USER_ID, memory)

#     # -----------------------------
#     # LOAD ALL MEMORIES
#     # -----------------------------
#     memories = memory_store.get_memories(USER_ID)

#     state["memory"] = memories

#     return state

#==================================================================================================================================================================================

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
    """
    # Get latest user message
    # Extract memory from the message using DeepSeek
    # Save new memory to SQLite
    # Load all saved memories from SQLite
    # Store them in state["memory"] 
    """

    print("\n" + "="*60)
    print("🆕 NEW QUESTION RECEIVED")
    print(f"QUESTION: {state['user_message']}")
    print("="*60)

    print("✅ memory_node executed")

    user_msg = state["user_message"] # Get the latest user message from state, to extract memory from it. This is the same "user_message" that was set in the initial state 
                                     # when the chat graph started executing in chat_ui.py. As the graph executes, this "user_message" can be updated with new user queries, 
                                     # and memory_node will always extract memory from the latest query.

    # -----------------------------
    # EXTRACT NEW MEMORY
    # -----------------------------
    memory = extract_memory(user_msg) # Send the latest user message to the DeepSeek model and get extracted memory from the response.
    #print(f"Extracted Memory: {memory}")
                                      
    if memory: # If the DeepSeek model extracted some memory (i.e. it's not None or empty), save that memory to the SQLite database using memory_store. 
        memory_store.add_memory(USER_ID, memory)

    # -----------------------------
    # LOAD ALL MEMORIES
    # -----------------------------
    memories = memory_store.get_memories(USER_ID)

    state["memory"] = memories # Save the list of all memories retrieved from the database into state["memory"]. 

    return state

