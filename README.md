# Pixel Loss Dashboard

A Streamlit application to visualize historic pixel-level losses for the Apr-Start pipeline.

## Prerequisites

Ensure you have the following Python packages installed:

```bash
pip install streamlit folium streamlit-folium altair pandas
```

## How to Run

1.  Navigate to the `filled_data` directory (if not already there).

2.  Run the application using Streamlit:

    ```bash
    streamlit run apr_start/web_app/app.py
    ```

    Or use the helper script:

    ```bash
    ./apr_start/web_app/run_app.sh
    ```

3.  The application should open automatically in your default web browser at `http://localhost:8501`.

## Features

-   **Interactive Map**: Explore total historic losses across Rio Grande do Sul.
-   **Pixel Detail**: Click a pixel to see a detailed breakdown of losses by year and window.
-   **Filters**: Filter by specific pixel IDs using the sidebar.
