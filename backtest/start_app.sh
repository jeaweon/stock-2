#!/usr/bin/env bash
# Move to script directory
cd "$(dirname "$0")"

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run Streamlit
streamlit run app.py
