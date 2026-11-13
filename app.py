"""Root Streamlit entrypoint forwarding cleanly to standalone paper_trading/app.py.

Allows existing Streamlit Cloud deployments configured with main module `app.py`
to launch the independent 1-hour live paper trading app without mixing backtest/ code.
"""
from pathlib import Path
import runpy

TARGET_APP = Path(__file__).resolve().parent / "paper_trading" / "app.py"
runpy.run_path(str(TARGET_APP), run_name="__main__")
