import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path().resolve().parent / '.src'))
from country_comparision import (
        run_anova, run_kruskal,print_country_metric_extremes
    )

from utils import (
                    report_na_and_duplicates_str,streamlit_boxplots,
                   streamlit_rolling_average,streamlit_hourly_trend,
                   streamlit_anomalies_plot,streamlit_mod_cleaning_plot,
                   scatter_plot_streamlit,corr_heatmap_streamlit,
                   plot_wind_rose_streamlit,plot_histograms_streamlit,
                   streamlit_avg_ghi_bar,streamlit_country_boxplots,
                   country_metric_extremes_str
                   )

country_file_map = {
    "Togo": "../data_for_deployment/togo-dapaong_qc_cleaned.csv",
    "Benin": "../data_for_deployment/benin_malanville_cleaned.csv",
    "Sierra Leone": "../data_for_deployment/sierraleone-bumbuna_cleaned.csv"
}
country = st.sidebar.selectbox(
    "Select Analysis Section",
    ["Cross-country Comparison","Togo", "Benin", "Sierra Leone"]
)
if country != "Cross-country Comparison":
    df_data = pd.read_csv(country_file_map[country], parse_dates=['Timestamp'])

    outlier_flag_cols = [col for col in df_data.columns if col.startswith('outlier_')]

    # Map True/False to "Outlier"/"Normal" for each of these columns
    for col in outlier_flag_cols:
        df_data[col] = df_data[col].map({True: "Outlier", False: "Normal"})

    st.header("Data Overview & Summary")
    st.markdown("""
    - **Summary statistics** for key columns are shown below.
    - **Missing value and duplicate reports** help assess data quality.
    """)
    if st.checkbox("Show data sample"):
        st.write(df_data.head())

    st.write("**Data Types:**")
    st.write(df_data.dtypes)

    st.write("**Summary Statistics:**")
    st.write(df_data.describe())

    # Display summary statistics for outlier flag columns
    report = report_na_and_duplicates_str(df_data)
    st.write("**Missing Values & Duplicates:**")
    st.text(report)

    st.header("Outlier Detection & Cleaning")
    st.markdown("""Outlier detection helps identify extreme values that may affect analysis. 
                Most variables show a few outliers, which are flagged for review.""")
    st.markdown("""
    - **Boxplots** below visualize outliers for key metrics.
    - Outlier flags are already included in the cleaned data and can be viewed below.
    """)

    target_cols = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust']
    #Function for plotting boxplots to highlight outliers
    streamlit_boxplots(df_data, target_cols)

    # Find and show outlier flag columns
    outlier_flag_cols = [col for col in df_data.columns if col.startswith('outlier_')]
    if outlier_flag_cols:
        st.info(f"Outlier flag columns detected: {', '.join(outlier_flag_cols)}")
        if st.checkbox("Show data with outlier flags"):
            st.write(df_data[['Timestamp'] + outlier_flag_cols].head())
            st.markdown("""
                        ### Outlier Detection in Columns
                        The columns contain values labeled as **"normal"** or **"outlier"**, indicating whether a specific metric is an outlier.  
                        For each column named **`outlier_{metric}`**, the corresponding value in the **`metric`** column determines whether it is classified as an outlier or a normal value.
                        """)
    else:
        st.warning("No outlier flag columns found in the data.")


    st.header("Time Series Analysis")
    st.markdown("""**Rolling averages** smooth out daily fluctuations,
                making it easier to spot long-term trends and 
                seasonal cycles in solar and temperature data.""")
    st.markdown("""
    - **Weekly rolling averages** reveal trends and seasonal patterns in the data.
    - Select a variable to visualize its trend over time.
    """)
    df_data["Timestamp_date"] = df_data['Timestamp'].dt.date
    df_data_date = df_data.groupby('Timestamp_date')[["GHI", "DNI", "DHI", "Tamb"]].mean()
    df_weekly = df_data_date.rolling(window=7).mean()

    var = st.selectbox("Select variable for rolling average", ["GHI", "DNI", "DHI", "Tamb"])
    color_map = {"GHI": "orange", "DNI": "red", "DHI": "blue", "Tamb": "green"}

    #Function for plotting rolling averages to highlight trends
    streamlit_rolling_average(df_weekly, var, color=color_map[var])


    st.header("Hourly Trend Analysis")
    st.markdown("""
    - **Hourly averages** show how solar and weather variables change throughout the day.
    - Select metrics to compare their daily patterns.
    """)
    df_data['hour'] = df_data['Timestamp'].dt.hour
    metrics = st.multiselect("Select metrics for hourly trend", 
                            ["GHI", "DNI", "DHI", "Tamb", "ModA", "ModB"],
                            default=["GHI", "DNI", "DHI"])


    #Function for plotting hourly trends to highlight daily patterns
    streamlit_hourly_trend(df_data, metrics)

    st.markdown("""### Solar and Wind Patterns

    - **Peak Irradiance:** Occurs midday (**10 AM–3 PM**).
    - **Temperature Lag:** Follows irradiance with a **1–2 hour delay**.
    - **Wind Speed Increase:** Rises in the afternoon due to **thermal convection**.""")


    st.header("Anomaly Detection")
    st.markdown("""
    - **Anomalies** (outliers) are highlighted for each metric.
    - Select a metric to view its anomaly trend.
    """)


    df_daily = df_data.groupby(df_data['Timestamp'].dt.date)[target_cols].mean().reset_index()
    df_daily['Timestamp'] = pd.to_datetime(df_daily['Timestamp'])
    z_scores = pd.DataFrame(stats.zscore(df_daily[target_cols]), columns=target_cols)
    for col in target_cols:
        df_daily[f'outlier_{col}'] = np.abs(z_scores[col]) > 3

    anomaly_metric = st.selectbox("Select metric for anomaly plot", target_cols)

    color_map2 = ['blue', 'Ivory ', 'green', 'orange', 'purple', 'brown', 'black']


    #Function for plotting anomalies to highlight outliers
    streamlit_anomalies_plot(df_daily, anomaly_metric, f"outlier_{anomaly_metric}", color=color_map2[target_cols.index(anomaly_metric)])
    st.markdown("""## Understanding Weather Anomalies

    Weather anomalies can serve as indicators of various phenomena, including:
    - **Sensor errors** that may affect data accuracy.
    - **Rare weather events** that deviate from typical patterns.
    - **General weather conditions** reflecting the region’s climate trends.

    Reviewing these anomalies is crucial for maintaining **data quality** and ensuring reliable weather analysis.""")


    st.header("Cleaning Impact")
    st.markdown("""
    - Visualize the effect of cleaning on module data.
    """)


    #Funtion for plotting cleaning impact to highlight the effect of cleaning on recorded radiance measurements
    streamlit_mod_cleaning_plot(df_data)
    st.markdown("""## Effect of Cleaning on Recorded Radiance Values

    The bar chart illustrates how **cleaning** impacts the **recorded radiance values**. Key insights include:
    - **Pre-cleaning values:** Radiance levels before cleaning, potentially affected by dust, debris, or sensor interference.
    - **Post-cleaning values:** Enhanced radiance measurements following cleaning, indicating improved sensor accuracy.
    - **Comparison:** A clear contrast between pre- and post-cleaning states, demonstrating the effectiveness of the cleaning process.

    This visualization helps in assessing the significance of maintenance procedures on data accuracy.""")



    st.header("Correlation & Relationship Analysis")
    st.markdown("""
    - **Correlation heatmaps** and **scatter plots** reveal relationships between variables.
    - Select variables to explore their interactions.
    """)
    corr_cols = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB']
    #Function for plotting correlation heatmaps to highlight relationships between variables
    corr_heatmap_streamlit(df_data, corr_cols)

    scatter_x = st.selectbox("Scatter plot X variable", ['WS', 'WSgust', 'WD', 'RH', 'Tamb', 'GHI'])
    scatter_y = st.selectbox("Scatter plot Y variable", ['GHI', 'Tamb', 'RH'])
    #Function for plotting scatter plots to highlight relationships between variables
    scatter_plot_streamlit(df_data, [scatter_x], scatter_y)

    st.markdown("""## Understanding Correlations in Data

    Strong **positive** or **negative** correlations indicate dependencies between variables. These relationships can be effectively visualized using **scatter plots**, which help in:
    - Identifying patterns and trends in data.
    - Assessing the strength and direction of correlations.
    - Understanding potential dependencies between variables.

    Scatter plots serve as a powerful tool for data analysis and interpretation.""")

    st.header("Wind & Distribution Analysis")
    st.markdown("""
    - **Wind rose** and **histograms** show wind patterns and variable distributions.
    """)

    #Function for plotting wind roses to highlight wind patterns and Histograms to highlight variable distributions
    plot_wind_rose_streamlit(df_data, ws_col='WS', wd_col='WD', bins=8)
    plot_histograms_streamlit(df_data, ['GHI', 'WS'], bins=30, kde=True, colors=['orange', 'blue'])
    st.markdown("""# Analyzing Wind and Solar Data

    Understanding atmospheric and solar dynamics is essential for weather analysis, energy production, and environmental studies. Various visualization methods help interpret key patterns:

    - **Wind Rose:** A graphical representation of prevailing wind directions and speeds. It provides insights into wind patterns, helping in applications such as meteorology, aviation, and renewable energy assessments.
    - **Histograms:** These statistical tools illustrate the distribution of solar and wind variables, offering a clearer view of variability, frequency, and trends over time.

    By leveraging these visualization techniques, researchers and analysts can gain deeper insights into weather conditions, climatological behavior, and the efficiency of renewable energy systems.""")


    st.header("Bubble Chart")
    st.markdown("""
    - Bubble chart visualizes the relationship between GHI, Tamb, and RH.
    """)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        df_data['GHI'], 
        df_data['Tamb'], 
        s=df_data['RH'],  
        alpha=0.5, 
        c=df_data['RH'],  
        cmap='viridis'
    )
    plt.xlabel('Global Horizontal Irradiance (GHI)')
    plt.ylabel('Ambient Temperature (Tamb)')
    plt.title('Bubble Chart: GHI vs. Tamb (Bubble Size = RH)')
    cbar = plt.colorbar()
    cbar.set_label('Relative Humidity (RH)')
    plt.tight_layout()
    st.pyplot(plt)

    st.markdown("""# Exploring Humidity and Its Interactions

    Data visualization plays a crucial role in analyzing atmospheric conditions. One effective method involves **bubble plots**, which provide intuitive insights into humidity variations and their relationships with other environmental factors.

    - **Bubble Size and Color:** Larger and darker bubbles represent **higher humidity levels**, making it easier to identify trends and variations.
    - **Interactions with Other Variables:** The plot highlights how **humidity interacts with temperature and solar irradiance**, offering valuable information for meteorological studies, climate research, and energy-related applications.

    By leveraging visual techniques like bubble plots, researchers can uncover patterns in humidity distribution and its broader impact on weather and environmental dynamics.""")


    st.header("Key Insights")
    st.markdown("""
    ### Summary & Conclusions

    - **Distinct Solar Patterns:** The region demonstrates pronounced daily and seasonal cycles in solar irradiance and temperature, with peak sunlight and energy potential occurring midday and during the dry season.
    - **Data Quality Assurance:** Systematic outlier and anomaly detection, along with cleaning procedures, have ensured that the dataset is robust and reliable for further analysis and modeling.
    - **Variable Interdependencies:** Correlation and scatter plot analyses reveal strong relationships between solar irradiance, temperature, and humidity. Notably, higher humidity often coincides with lower irradiance and temperature, highlighting the impact of atmospheric moisture on solar resource availability.
    - **Wind and Environmental Dynamics:** Wind rose and histogram analyses show prevailing wind directions and speeds, which are crucial for both meteorological understanding and renewable energy planning. Wind patterns also influence temperature and humidity distributions.
    - **Actionable Insights:** The comprehensive EDA provides a solid foundation for site assessment, energy forecasting, and climate research. The interactive dashboard enables stakeholders to explore specific variables, identify trends, and make informed decisions based on high-quality data.

    ---
    This dashboard empowers users to interactively explore and interpret the region's solar and meteorological data, supporting data-driven strategies for energy and environmental applications.
    """)

else:
    st.title("Cross-country Solar Resource Comparison")
    st.markdown("""
    This section compares solar resource metrics across **Benin**, **Togo**, and **Sierra Leone** using summary statistics, boxplots, and statistical tests.
    """)

    # Load data
    benin_df = pd.read_csv("../data_for_deployment/benin_malanville_cleaned.csv")
    togo_df = pd.read_csv("../data_for_deployment/togo-dapaong_qc_cleaned.csv")
    sierraleone_df = pd.read_csv("../data_for_deployment/sierraleone-bumbuna_cleaned.csv")
    benin_df['Country'] = 'Benin'
    togo_df['Country'] = 'Togo'
    sierraleone_df['Country'] = 'Sierra Leone'
    combined_df = pd.concat([benin_df, togo_df, sierraleone_df], ignore_index=True)
    combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'])

    # --- Metric Comparison ---
    st.header("Metric Comparison: Boxplots")
    st.markdown("""
    Boxplots below visualize the distribution and variability of **GHI**, **DNI**, and **DHI** for each country.
    """)
    metric = st.selectbox("Select metric for boxplot", ["GHI", "DNI", "DHI"])
    #Function for plotting boxplots to highlight distribution and variability of GHI, DNI, and DHI
    streamlit_country_boxplots(combined_df, [metric])

    # --- Detailed Markdown for Each Metric ---
    if metric == "GHI":
        st.markdown("""
        **Analysis of GHI Boxplots (Benin, Togo, Sierra Leone)**  
        - **Benin:** Largest spread and highest central variability; most reliable high solar potential; stable minimum GHI.
        - **Togo:** Moderate variability; unpredictable high-energy events; stable low-end GHI.
        - **Sierra Leone:** Lowest central variability; frequent erratic highs; stable minimum GHI.
        - **Conclusion:** Benin is optimal for baseline solar energy; Togo needs storage; Sierra Leone benefits from hybrid systems.
        """)
    elif metric == "DNI":
        st.markdown("""
        **Analysis of DNI Boxplots (Benin, Togo, Sierra Leone)**  
        - **Benin:** Highest central variability and reliable high DNI; stable minimum.
        - **Togo:** Balanced variability; highest outlier spikes; stable minimum.
        - **Sierra Leone:** Tight clustering; frequent extreme spikes; stable minimum.
        - **Conclusion:** Benin best for large-scale solar; Togo for moderate projects with storage; Sierra Leone for hybrid systems.
        """)
    elif metric == "DHI":
        st.markdown("""
        **Analysis of DHI Boxplots (Benin, Togo, Sierra Leone)**  
        - **Benin/Togo:** Similar variability; reliable high DHI; stable minimum.
        - **Sierra Leone:** Slightly higher variability and more extreme spikes.
        - **Conclusion:** All regions have dependable low-end DHI; Sierra Leone may benefit from hybrid systems.
        """)

    # --- Summary Table ---
    st.header("Summary Table")
    st.markdown("""
    The table below summarizes the mean, median, and standard deviation for each metric by country.
    """)
    summary = combined_df.groupby('Country')[['GHI', 'DNI', 'DHI']].agg(['mean', 'median', 'std'])
    st.dataframe(summary)
    st.markdown("""
    The following highlights which country has the highest and lowest values for each metric/statistic:
    """)
    #function for printing country metric extremes
    st.markdown(country_metric_extremes_str(summary))


    # --- Statistical Analysis ---
    st.header("Statistical Analysis")
    st.markdown("""
    Statistical tests are used to determine if the differences in GHI between countries are significant.
    - **One-way ANOVA** (parametric)
    - **Kruskal–Wallis** (non-parametric)
    """)

    p_anova = run_anova(benin_df['GHI'], togo_df['GHI'], sierraleone_df['GHI'])
    p_kruskal = run_kruskal(benin_df['GHI'], togo_df['GHI'], sierraleone_df['GHI'])
    p_anova_bt = run_anova(benin_df['GHI'], togo_df['GHI'])
    p_kruskal_bt = run_kruskal(benin_df['GHI'], togo_df['GHI'])
    p_anova_bs = run_anova(benin_df['GHI'], sierraleone_df['GHI'])
    p_kruskal_bs = run_kruskal(benin_df['GHI'], sierraleone_df['GHI'])
    p_anova_ts = run_anova(togo_df['GHI'], sierraleone_df['GHI'])
    p_kruskal_ts = run_kruskal(togo_df['GHI'], sierraleone_df['GHI'])

    st.markdown(f"""
    **ANOVA p-value (all):** {p_anova:.5g}  
    **Kruskal–Wallis p-value (all):** {p_kruskal:.5g}  
    **ANOVA p-value (Benin vs Togo):** {p_anova_bt:.5g}  
    **Kruskal–Wallis p-value (Benin vs Togo):** {p_kruskal_bt:.5g}  
    **ANOVA p-value (Benin vs Sierra Leone):** {p_anova_bs:.5g}  
    **Kruskal–Wallis p-value (Benin vs Sierra Leone):** {p_kruskal_bs:.5g}  
    **ANOVA p-value (Togo vs Sierra Leone):** {p_anova_ts:.5g}  
    **Kruskal–Wallis p-value (Togo vs Sierra Leone):** {p_kruskal_ts:.5g}
    """)

    st.markdown("""
    #### Key Findings  
    - **All countries differ significantly in GHI** (ANOVA/Kruskal–Wallis: *p ≈ 0*).  
    - **Benin vs. Togo/Sierra Leone** and **Togo vs. Sierra Leone** show *p ≈ 0* → **GHI is distinct regionally**.  
    - **Benin** likely has the **highest GHI**, but pairwise tests confirm **all differences are near-certain (not random)**.  
    """)

    st.markdown("""
    ### Key Observations 
    - **Benin** exhibits the **highest mean and median GHI**, confirming it as the strongest solar resource, but with the **greatest variability** (widest IQR), indicating fluctuating solar output.  
    - **Sierra Leone** has the **lowest median and mean GHI**, yet shows the **most extreme outlier spikes** (highest recorded values), suggesting rare but intense solar events.  
    - **Statistical tests (ANOVA/Kruskal–Wallis)** confirm *significant differences* (p ≈ 0) between all country pairs, underscoring **region-specific solar profiles**: Benin (high potential, variable), Togo (moderate), Sierra Leone (low baseline, erratic spikes).  
    """)

    # --- Visual Summary ---
    st.header("Visual Summary")
    st.markdown("""
    The bar chart below ranks countries by average GHI.
    """)
    #Function for plotting bar charts to highlight average GHI
    streamlit_avg_ghi_bar(combined_df)

    st.markdown("""---

### Cross-country Analysis Summary & Insights

- **Benin** stands out as the region with the highest average and median GHI, making it the most promising for consistent solar energy generation. However, its greater variability suggests that solar output can fluctuate, highlighting the need for robust energy storage or grid integration solutions.
- **Togo** demonstrates moderate solar potential with less variability than Benin, offering a balance between reliability and output. This makes Togo suitable for solar projects where steady performance is valued, though occasional high-energy events may still occur.
- **Sierra Leone** has the lowest average and median GHI, but exhibits the most extreme outlier spikes. This indicates that while baseline solar resource is lower, there are rare periods of intense solar irradiance, which could be leveraged with hybrid or flexible energy systems.
- **Statistical tests** confirm that the differences in solar resource distributions between all three countries are highly significant, emphasizing the importance of region-specific planning and technology selection for solar energy deployment.

**Conclusion:**  
The cross-country analysis reveals that solar resource characteristics are distinctly different across Benin, Togo, and Sierra Leone. These insights can guide policymakers, developers, and researchers in optimizing solar energy strategies tailored to each region’s unique climate and variability 
profile. The interactive dashboard enables ongoing exploration and comparison, supporting data-driven decisions for sustainable energy development in West Africa.""")