#!/usr/bin/env bash
cd "$(dirname "$0")"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

streamlit run paper_trading_app.py
