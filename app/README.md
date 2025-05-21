# Streamlit Solar Data Dashboard (`app/main.py`)

This file contains the main code for the interactive Streamlit dashboard for the **solar-challenge-week1** project.  
The dashboard enables users to explore, analyze, and compare solar and meteorological data for three regions: **Togo**, **Benin**, and **Sierra Leone**.

---

## Features

- **Sidebar Navigation:**  
  Users can select a region or a cross-country comparison from the sidebar.

- **Data Overview & Summary:**  
  View summary statistics, data types, and missing/duplicate value reports for the selected region.

- **Outlier Detection & Cleaning:**  
  Visualize outliers with boxplots and inspect outlier flags already present in the cleaned data.

- **Time Series Analysis:**  
  Explore weekly rolling averages for key variables to identify trends and seasonal patterns.

- **Hourly Trend Analysis:**  
  Analyze how solar and meteorological variables change throughout the day.

- **Anomaly Detection:**  
  Detect and visualize anomalies in the data using Z-score-based outlier detection.

- **Cleaning Impact:**  
  Assess the effect of cleaning on module data through comparative plots.

- **Correlation & Relationship Analysis:**  
  Examine correlation heatmaps and scatter plots to understand dependencies between variables.

- **Wind & Distribution Analysis:**  
  Visualize wind patterns with wind rose plots and explore variable distributions with histograms.

- **Bubble Chart:**  
  Interactive bubble plots to explore the relationship between GHI, temperature, and humidity.

- **Key Insights:**  
  Summarized findings and actionable insights for each region and for cross-country comparison.

- **Cross-country Comparison:**  
  Compare GHI, DNI, and DHI across all regions using boxplots, summary tables, statistical tests (ANOVA, Kruskal–Wallis), and visual summaries.

---

## About `utils.py`

The `utils.py` file in the `app` directory contains helper functions specifically designed for Streamlit integration, including:

- **report_na_and_duplicates_str:**  
  Returns a string summary of missing values and duplicates for display in Streamlit.
- **streamlit_boxplots, streamlit_rolling_average, streamlit_hourly_trend, streamlit_anomalies_plot, streamlit_mod_cleaning_plot, scatter_plot_streamlit, corr_heatmap_streamlit, plot_wind_rose_streamlit, plot_histograms_streamlit, streamlit_avg_ghi_bar, streamlit_country_boxplots:**  
  Streamlit-compatible plotting functions that render interactive visualizations directly in the dashboard.

These utilities ensure a smooth, interactive, and user-friendly experience for dashboard users.

---

## Usage Instructions

1. **Install dependencies:**  
   Make sure you have all required packages installed (see `requirements.txt`).  
   Activate your virtual environment if you use one.

   ```sh
   pip install -r requirements.txt