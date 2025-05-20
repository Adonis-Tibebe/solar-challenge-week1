# Source Code (`src`)

This folder contains the core Python modules used for data analysis, visualization, and statistical comparison in the project.

### Main modules

- **plots.py**  
  Utility functions for generating plots and visualizations, including time series, correlation heatmaps, scatter plots, wind roses, histograms, and cross-country comparison bar charts.

- **profiling_and_cleaning.py**  
  Functions for data profiling, missing value and duplicate detection, outlier detection, and data cleaning.

- **country_comparision.py**  
  Modular functions for comparing solar metrics across countries. Includes:
  - Boxplot generation for multiple metrics by country
  - Statistical tests (ANOVA and Kruskal–Wallis) for overall and pairwise group comparisons
  - Summary reporting of which country has the highest and lowest mean, median, and standard deviation for each metric
  - Visual summaries such as ranking countries by average GHI

### Usage

These modules are imported and used in the Jupyter notebooks for consistent, reusable, and robust analysis across all regions and for cross-country comparisons.  
You can also use them in your own scripts for similar solar data analysis and statistical tasks.