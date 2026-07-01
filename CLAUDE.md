# Python Environment

This project uses the virtual environment `.venv2`.

Always use:

.venv2\Scripts\python.exe

Examples:

.venv2\Scripts\python.exe -m pip install -r requirements.txt

.venv2\Scripts\python.exe -m streamlit run streamlit_app/main.py

.venv2\Scripts\python.exe -m uvicorn app:app --reload --port 8000