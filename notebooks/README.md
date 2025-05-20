# Solar Resource Analysis Notebooks  
**Exploratory Data Analysis (EDA) for Benin, Togo, and Sierra Leone**  

---

## Notebooks Included  
- `benin_eda.ipynb`: EDA for **Benin** solar dataset.  
- `togo_eda.ipynb`: EDA for **Togo** solar dataset.  
- `sierraleone_eda.ipynb`: EDA for **Sierra Leone** solar dataset.  
- `compare_countries.ipynb`: **Cross-country comparison and statistical analysis**.  

---

## What You'll Find in Each Notebook  
- **Data Cleaning**: Missing value reports, outlier detection, and preprocessing.  
- **Time Series Analysis**: Hourly/daily trends for solar metrics (GHI, DNI, DHI).  
- **Correlation Analysis**: Relationships between solar irradiance, temperature, and wind.  
- **Visualizations**:  
  - Distribution plots (wind, temperature).  
  - Bubble charts and regression plots.  

---

## Cross-Country Comparison (`compare_countries.ipynb`)  
- **Boxplots**: Compare GHI, DNI, and DHI distributions across countries.  
- **Statistical Tests**:  
  - ANOVA and Kruskal–Wallis tests for significant differences.  
  - Pairwise comparisons (Benin vs. Togo, Benin vs. Sierra Leone, etc.).  
- **Summary Tables**:  
  - Highest/lowest mean, median, and variability for solar metrics.  
- **Ranking Charts**: Visual bar charts for country performance (e.g., average GHI).  

---

## Key Findings & Observations  
- **Benin**: Highest median solar potential but greatest variability.  
- **Sierra Leone**: Lowest baseline irradiance but extreme outlier spikes.  
- **Togo**: Moderate solar resource with balanced stability.  
- **Statistical Significance**: All country pairs show *p ≈ 0* (irradiance differs regionally).  

---

## Purpose  
These notebooks enable **interactive exploration** of solar datasets, robust statistical comparisons, and actionable insights for solar energy deployment.  