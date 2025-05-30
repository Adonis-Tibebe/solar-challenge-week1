> **Note:**  
> For optimal performance and to prevent timeouts on Streamlit Community Cloud, all visualizations in the dashboard use downsampled data. Large datasets are automatically sampled before plotting to ensure smooth and responsive user experience.(extended delays were experienced otherwise)
> here is the link below:
https://solar-challenge-week1-2w82gxwrdkdqmumt8ba2vq.streamlit.app/

# Solar Challenge Week 0

This repository contains a robust, reproducible workflow for solar data analysis across Benin, Togo, and Sierra Leone. The project covers data profiling, cleaning, exploratory data analysis (EDA), cross-country comparison, and an interactive dashboard for visualization and insight generation.

---

## Project Structure

- `notebooks/` — Jupyter notebooks for EDA and cross-country comparison
- `src/` — Reusable Python modules for data analysis, cleaning, and visualization
- `app/` — Streamlit dashboard application and utilities
- `tests/` — Automated test scripts using pytest
- `scripts/` — (For future automation)
- `.github/workflows/` — CI/CD configuration
- `.vscode/` — Editor settings
- `requirements.txt` — Python dependencies
- `README.md` — Project overview and instructions

---

## How to Reproduce the Environment

1. **Clone the repo**
2. **Create a virtual environment:**
   ```sh
   python -m venv venv
