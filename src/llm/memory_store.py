# ===============================
# memory_store.py - SQLITE MEMORY STORE (NO SERVER NEEDED)
# ===============================
import sqlite3
import uuid


class SQLiteMemoryStore:

    def __init__(self, db_path="memory.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS user_memory (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            memory TEXT
        )
        """)
        self.conn.commit()

    def get_memories(self, user_id):
        """This function loads previous user memories so our chatbot can remember preferences"""
        cur = self.conn.cursor()  # Create cursor object to execute SQL queries on the database
        cur.execute("SELECT memory FROM user_memory WHERE user_id=?", (user_id,)) #Database returns matching rows
        return [row[0] for row in cur.fetchall()]

    def add_memory(self, user_id, text):
        existing = self.get_memories(user_id)

        if text.lower() in [e.lower() for e in existing]:
            return

        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO user_memory VALUES (?, ?, ?)",
            (str(uuid.uuid4()), user_id, text) #uuid creates unique memory row IDs, NOT user IDs. user_id we have define already as -
                                               #USER_ID = "default_user" as static user_id in memory_node.py
                                               #and one user can have multiple memories, so we need unique IDs for each memory row. UUID is used to create that unique memory ID.
        )
        self.conn.commit()


