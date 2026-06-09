#================================
# estatemind_copilot.py
#================================

import streamlit as st
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.ui.intent_chat_ui import render_intent_chat_workspace

st.set_page_config(
    page_title="EstateMind Copilot",
    page_icon="🧠",
    layout="wide"
)

render_intent_chat_workspace()