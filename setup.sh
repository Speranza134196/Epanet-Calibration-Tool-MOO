#!/bin/bash
pip install -r requirements.txt
streamlit run web_app.py --server.port $PORT --server.enableCORS false
