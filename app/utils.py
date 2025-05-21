import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Helper: Downsample DataFrame for plotting if too large
def _downsample_df(df, max_rows=5000):
    """Downsample DataFrame for plotting if too large."""
    if len(df) > max_rows:
        return df.sample(max_rows, random_state=42)
    return df

def country_metric_extremes_str(summary, metrics=['GHI', 'DNI', 'DHI'], stats=['mean', 'median', 'std']):
    """
    Returns a Markdown-formatted string summarizing the country with the highest and lowest value
    for each metric and statistic, suitable for display in Streamlit.
    """
    output = []
    for metric in metrics:
        output.append(f"**{metric}**")
        for stat in stats:
            values = summary[(metric, stat)]
            max_country = values.idxmax()
            min_country = values.idxmin()
            max_value = values.max()
            min_value = values.min()
            output.append(f"- {stat.capitalize()}:  \n"
                          f"&nbsp;&nbsp;• Highest: **{max_country}** ({max_value:.2f})  \n"
                          f"&nbsp;&nbsp;• Lowest: **{min_country}** ({min_value:.2f})")
        output.append("")  # Blank line for spacing
    return "\n".join(output)

def streamlit_country_boxplots(df, metrics, country_col='Country', palette=None, figsize=(10, 5)):
    """
    Streamlit + Plotly version: Interactive boxplots for specified metrics grouped by country.
    Adds error spotting for empty data, missing columns, and plotting errors.
    """
    df = _downsample_df(df)
    if df.empty:
        st.warning("No data available for plotting after downsampling/filtering.")
        return
    if palette is None:
        unique_countries = df[country_col].unique()
        palette = dict(zip(unique_countries, sns.color_palette("tab10", len(unique_countries)).as_hex()))

    for metric in metrics:
        if metric not in df.columns:
            st.error(f"Column '{metric}' not found in data.")
            continue
        if df[metric].dropna().empty:
            st.warning(f"No valid data for {metric}.")
            continue
        try:
            st.subheader(f"Boxplot of {metric} by {country_col}")
            fig = px.box(
                df,
                x=country_col,
                y=metric,
                color=country_col,
                color_discrete_map=palette,
                points="outliers",
                title=f"Boxplot of {metric} by {country_col}",
            )
            fig.update_layout(
                xaxis_title=country_col,
                yaxis_title=metric,
                height=500,
                margin=dict(t=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Plotting failed for {metric}: {e}")

def streamlit_avg_ghi_bar(df, country_col='Country', ghi_col='GHI'):
    """
    Plots a bar chart ranking countries by average GHI using Streamlit (interactive version).
    """
    df = _downsample_df(df)
    avg_ghi = df.groupby(country_col)[ghi_col].mean().sort_values(ascending=False).reset_index()

    fig = px.bar(
        avg_ghi,
        x=country_col,
        y=ghi_col,
        color=country_col,
        text_auto='.2s',
        title='Average GHI by Country',
        labels={ghi_col: 'Average GHI', country_col: 'Country'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(
        xaxis_tickangle=45,
        height=500,
        margin=dict(t=40, b=40),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_histograms_streamlit(df, cols, bins=30, kde=True, colors=None):
    """
    Interactive histograms.
    """
    df = _downsample_df(df)
    n = len(cols)
    if colors is None:
        colors = sns.color_palette("husl", n).as_hex()

    for i, col in enumerate(cols):
        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(
            x=df[col],
            nbinsx=bins,
            marker_color=colors[i],
            opacity=0.75,
            name=col
        ))

        # Optional KDE line
        if kde:
            from scipy.stats import gaussian_kde
            x_range = np.linspace(df[col].min(), df[col].max(), 200)
            kde_y = gaussian_kde(df[col].dropna())(x_range)
            kde_y_scaled = kde_y * len(df[col]) * (x_range[1] - x_range[0])  # scale to histogram area

            fig.add_trace(go.Scatter(
                x=x_range,
                y=kde_y_scaled,
                mode='lines',
                name='KDE',
                line=dict(color='black')
            ))

        fig.update_layout(
            title=f'Histogram of {col}',
            xaxis_title=col,
            yaxis_title='Frequency',
            bargap=0.1,
            height=400
        )

        st.subheader(f'Histogram of {col}')
        st.plotly_chart(fig, use_container_width=True)

def plot_wind_rose_streamlit(df, ws_col='WS', wd_col='WD', bins=8):
    """
    Interactive Wind Rose plot.
    """
    df = _downsample_df(df)
    wd = df[wd_col] % 360
    ws = df[ws_col]
    wd_bins = np.linspace(0, 360, bins + 1)
    wd_centers = (wd_bins[:-1] + wd_bins[1:]) / 2

    ws_means = []
    for i in range(bins):
        mask = (wd >= wd_bins[i]) & (wd < wd_bins[i + 1])
        ws_means.append(ws[mask].mean() if mask.any() else 0)

    # Close the circle
    ws_means.append(ws_means[0])
    wd_centers = np.append(wd_centers, wd_centers[0])

    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=ws_means,
        theta=wd_centers,
        width=[360/bins] * len(ws_means),
        marker_color='skyblue',
        marker_line_color='black',
        marker_line_width=1,
        opacity=0.75
    ))

    fig.update_layout(
        polar=dict(
            angularaxis=dict(
                direction='clockwise',
                rotation=90,
                tickmode='array',
                tickvals=np.linspace(0, 360, bins, endpoint=False),
                ticktext=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][:bins]
            ),
            radialaxis=dict(visible=True)
        ),
        showlegend=False,
        height=600,
        margin=dict(t=40, b=20)
    )

    st.subheader("Wind Rose (WS/WD)")
    st.plotly_chart(fig, use_container_width=True)

def corr_heatmap_streamlit(df, cols, title="Correlation Heatmap"):
    """
    Streamlit + Plotly version: Interactive correlation heatmap.
    """
    df = _downsample_df(df)
    # Compute correlation matrix
    corr_matrix = df[cols].corr().round(2)

    z = corr_matrix.values
    x = corr_matrix.columns.tolist()
    y = corr_matrix.index.tolist()

    # Create heatmap trace
    heatmap = go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Correlation'),
        hovertemplate='Corr(%{x}, %{y}) = %{z}<extra></extra>'
    )

    # Add text annotations
    annotations = []
    for i in range(len(y)):
        for j in range(len(x)):
            annotations.append(
                dict(
                    x=x[j],
                    y=y[i],
                    text=str(z[i][j]),
                    showarrow=False,
                    font=dict(color='black' if abs(z[i][j]) < 0.7 else 'white'),
                    xanchor='center',
                    yanchor='middle'
                )
            )

    layout = go.Layout(
        annotations=annotations,
        xaxis=dict(tickangle=45),
        margin=dict(l=40, r=40, t=20, b=40),
        height=500
    )

    fig = go.Figure(data=[heatmap], layout=layout)

    # Streamlit UI
    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True)

def scatter_plot_streamlit(df, cols, base_col, alpha=0.5):
    """
    Scatter plots of each col in cols vs. base_col.
    """
    df = _downsample_df(df)
    for col in cols:
        trace = go.Scatter(
            x=df[base_col],
            y=df[col],
            mode='markers',
            marker=dict(opacity=alpha, size=6),
            name=f'{col} vs. {base_col}'
        )

        layout = go.Layout(
            xaxis=dict(title=base_col),
            yaxis=dict(title=col),
            margin=dict(l=40, r=40, t=20, b=40),
            height=400
        )

        fig = go.Figure(data=[trace], layout=layout)

        # Streamlit UI
        st.subheader(f'{col} vs. {base_col}')
        st.plotly_chart(fig, use_container_width=True)

def streamlit_mod_cleaning_plot(df, cleaning_col='Cleaning'):
    """
    Interactive bar chart of average ModA & ModB for pre/post-clean.
    """
    df = _downsample_df(df)
    # Group by cleaning status and calculate mean ModA & ModB
    mod_means = df.groupby(cleaning_col)[['ModA', 'ModB']].mean().reset_index()
    mod_means['Status'] = mod_means[cleaning_col].map({0: 'Pre-clean', 1: 'Post-clean'})

    # X axis categories
    x = mod_means['Status']

    # Bar traces
    trace_modA = go.Bar(x=x, y=mod_means['ModA'], name='ModA')
    trace_modB = go.Bar(x=x, y=mod_means['ModB'], name='ModB')

    # Layout settings
    layout = go.Layout(
        barmode='group',  # Grouped bars
        xaxis=dict(title='Cleaning Status'),  # X axis title
        yaxis=dict(title='Average Value'),  # Y axis title
        margin=dict(l=40, r=40, t=20, b=40),  # Margin padding
        height=400  # Plot height
    )

    # Create the figure
    fig = go.Figure(data=[trace_modA, trace_modB], layout=layout)

    # Title as subheader
    st.subheader('Average ModA & ModB: Pre-clean vs Post-clean')

    # Show interactive plot
    st.plotly_chart(fig, use_container_width=True)

def streamlit_anomalies_plot(df, value_col, outlier_col, title=None, ylabel=None, color='blue'):
    """
    Interactive time series with anomaly markers.
    """
    df = _downsample_df(df)
    # Create base line trace
    trace_main = go.Scatter(
        x=df['Timestamp'],
        y=df[value_col],
        mode='lines',
        name=value_col,
        line=dict(color=color)
    )

    # Create anomaly scatter trace
    trace_anomalies = go.Scatter(
        x=df[df[outlier_col]]['Timestamp'],
        y=df[df[outlier_col]][value_col],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=8, symbol='circle')
    )

    # Combine and define layout
    layout = go.Layout(
        title=None,  # We use Streamlit's subheader for title
        xaxis=dict(title='Date'),
        yaxis=dict(title=ylabel or value_col),
        legend=dict(x=0, y=1),
        margin=dict(l=40, r=40, t=20, b=40),
        height=500
    )

    fig = go.Figure(data=[trace_main, trace_anomalies], layout=layout)

    # Title as subheader
    st.subheader(title or f"{value_col} with Highlighted Anomalies")

    # Show interactive plot
    st.plotly_chart(fig, use_container_width=True)

def streamlit_hourly_trend(df, value_cols, xlabel='Hour of Day', ylabel='Mean Value'):
    """
    Creates an interactive bar chart showing the hourly mean trend for specified columns using Plotly in Streamlit.
    """
    df = _downsample_df(df)
    st.subheader("Hourly Trend of Selected Metrics")

    # Check if 'hour' column exists
    if 'hour' not in df.columns:
        st.error("The DataFrame must contain an 'hour' column.")
        return

    # Check if value columns exist in the DataFrame
    missing_cols = [col for col in value_cols if col not in df.columns]
    if missing_cols:
        st.error(f"The following columns are missing from the DataFrame: {', '.join(missing_cols)}")
        return

    # Group by 'hour' and calculate the mean for specified columns
    hourly_mean = df.groupby('hour')[value_cols].mean().reset_index()

    # Melt the DataFrame for easy plotting
    melted_df = hourly_mean.melt(id_vars='hour', var_name='Metric', value_name='Mean Value')

    # Create the interactive bar chart using Plotly
    fig = px.bar(
        melted_df,
        x='hour',
        y='Mean Value',
        color='Metric',
        barmode='group',
        title=None,  # Title is handled by Streamlit's subheader
        labels={'hour': xlabel, 'Mean Value': ylabel},
        template='plotly_white'
    )

    # Customize the layout
    fig.update_layout(
        xaxis=dict(tickmode='linear'),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        legend_title="Metrics",
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)

def streamlit_rolling_average(df, column_name, title=None, ylabel=None, color='orange'):
    """
    Creates an interactive 7-day rolling average line chart for a specific column using Plotly in Streamlit.
    """
    df = _downsample_df(df)
    st.subheader("Interactive Rolling Average Line Chart")

    # Check if the selected column exists in the DataFrame
    if column_name not in df.columns:
        st.error(f"Column '{column_name}' does not exist in the DataFrame.")
        return

    # Create the Plotly figure
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[column_name],
            mode="lines",
            name=f"7-Day Rolling Avg of {column_name}",
            line=dict(color=color, width=2)
        )
    )

    # Add layout and styling
    fig.update_layout(
        title=title or f"7-Day Rolling Average of {column_name}",
        xaxis_title="Date",
        yaxis_title=ylabel or column_name,
        template="plotly_white",
        xaxis=dict(showgrid=True, tickangle=45),
        yaxis=dict(showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)

def streamlit_boxplots(df, cols):
    """
    Creates interactive boxplots for the specified columns using Plotly in Streamlit.
    """
    df = _downsample_df(df)
    st.subheader("Interactive Boxplots for Outlier Detection")

    # Sidebar for column selection
    selected_cols = st.multiselect(
        "Select columns to plot:",
        cols,
        default=cols  # By default, all columns are selected
    )

    # Check if columns are selected
    if not selected_cols:
        st.warning("Please select at least one column to visualize.")
        return

    # Melt the DataFrame for easy plotting with Plotly
    melted_df = df[selected_cols].melt(var_name="Variable", value_name="Value")

    # Plot the boxplot using Plotly
    fig = px.box(
        melted_df,
        x="Variable",
        y="Value",
        color="Variable",
        title="Interactive Boxplots for Outlier Detection",
        labels={"Value": "Values", "Variable": "Columns"},
        template="plotly_white"
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

def report_na_and_duplicates_str(df: pd.DataFrame) -> str:
    """
    Returns a string summary of missing values and duplicates in the DataFrame,
    suitable for display in Streamlit.
    """
    output = []
    na_counts = df.isna().sum()
    na_cols = na_counts[na_counts > 0]
    if not na_cols.empty:
        output.append("Missing values by column:")
        for col, count in na_cols.items():
            output.append(f"  - {col}: {count}")
    else:
        output.append("No missing values found.")

    dup_count = df.duplicated().sum()
    if dup_count > 0:
        output.append(f"Number of duplicate rows: {dup_count}")
    else:
        output.append("No duplicate rows found.")

    return "\n".join(output)