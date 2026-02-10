#!/bin/bash
# Run the Pixel Loss Dashboard

# Ensure we are in the script's directory or handle paths correctly
cd "$(dirname "$0")"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Error: streamlit is not installed."
    echo "Please install it with: pip install streamlit folium streamlit-folium altair pandas"
    exit 1
fi

echo "Starting Pixel Loss Dashboard..."
streamlit run app.py
