# ================================
# estatemind_copilot.py
# ================================

import sys

# Configure UTF-8 output for Windows (prevents UnicodeEncodeError on emoji prints)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import streamlit as st
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