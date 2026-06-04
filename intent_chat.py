# =====================================================================
# intent_chat.py (Root Level Wrapper File)
# =====================================================================

import sys
from pathlib import Path
import streamlit as st

# Ensure root directory is mapped correctly for clean module imports
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.ui.intent_chat_ui import render_intent_chat_workspace

st.set_page_config(layout="wide", page_title="AI Intent Chatbot Workspace")

# Run the UI workspace natively
render_intent_chat_workspace()

