import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns


# Date formatters (make sure these are defined before using the function)
weeks = mdates.WeekdayLocator(byweekday=mdates.MO)
months = mdates.MonthLocator()
years_fmt = mdates.DateFormatter("%B %Y")

def plot_rolling_average(df, column_name, title=None, ylabel=None, color='orange'):
    """
    Plots a 7-day rolling average line chart for a specific column from a given DataFrame.
    
    Parameters:
        df (DataFrame): The weekly-averaged DataFrame (e.g., df_weekly)
        column_name (str): Column to be plotted (must exist in df)
        title (str): Plot title (optional)
        ylabel (str): Y-axis label (optional)
        color (str): Line color (optional)
    """
    plt.figure(figsize=(15, 8), dpi=300)
    plt.xticks(fontsize=10, rotation=45)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.plot(df.index, df[column_name], label=f'Weekly {column_name} (Rolling Avg)', color=color)
    
    plt.gca().set_xlim(df.index.min(), df.index.max())
    
    # Date formatting
    plt.gca().xaxis.set_major_locator(months)
    plt.gca().xaxis.set_minor_locator(weeks)
    plt.gca().xaxis.set_major_formatter(years_fmt)
    
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel(ylabel or column_name)
    plt.title(title or f'7-Day Rolling Average of {column_name}')
    plt.tight_layout()
    plt.show()


def plot_hourly_trend(df, value_cols, title='Hourly Trend', xlabel='Hour of Day', ylabel='Mean Value', figsize=(15,8)):
    """
    Plots the hourly mean trend for specified columns in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing an 'hour' column and value columns.
        value_cols (list): List of column names to plot.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        figsize (tuple): Figure size.
    """
    hourly_mean = df.groupby('hour')[value_cols].mean()
    ax = hourly_mean.plot(kind='bar', figsize=figsize)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_anomalies(df, value_col, outlier_col, title=None, ylabel=None, color='blue'):
    """
    Plots a time series with anomalies highlighted as scatter points.
    
    Parameters:
        df (DataFrame): The full DataFrame with Timestamp, value_col, and outlier_col.
        value_col (str): The name of the column to plot.
        outlier_col (str): The name of the Boolean column indicating outliers.
        title (str): Optional plot title.
        ylabel (str): Optional label for the y-axis.
        color (str): Line color for the base time series.
    """
    plt.figure(figsize=(15, 8), dpi=300)
    plt.xticks(fontsize=10, rotation=45)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Plot the full time series
    plt.plot(df['Timestamp'], df[value_col], label=value_col, color=color)
    
    # Highlight anomalies using red scatter points
    plt.scatter(df[df[outlier_col]]['Timestamp'],
                df[df[outlier_col]][value_col],
                color='red', label='Anomaly', zorder=5)
    
    # Axis formatting
    plt.gca().set_xlim(df['Timestamp'].min(), df['Timestamp'].max())
    plt.gca().xaxis.set_major_locator(months)
    plt.gca().xaxis.set_minor_locator(weeks)
    plt.gca().xaxis.set_major_formatter(years_fmt)
    
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel(ylabel or value_col)
    plt.title(title or f"{value_col} with Highlighted Anomalies")
    plt.tight_layout()
    plt.show()



def plot_mod_cleaning(df, cleaning_col='Cleaning'):
    """
    Plots average ModA & ModB for pre-clean and post-clean groups.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        cleaning_col (str): The column name indicating cleaning status (0=pre, 1=post).
    """

    # Group and calculate means
    mod_means = df.groupby(cleaning_col)[['ModA', 'ModB']].mean().reset_index()
    mod_means['Status'] = mod_means[cleaning_col].map({0: 'Pre-clean', 1: 'Post-clean'})

    # Plot
    plt.figure(figsize=(8, 5))
    bar_width = 0.35
    x = np.arange(len(mod_means['Status']))

    plt.bar(x - bar_width/2, mod_means['ModA'], width=bar_width, label='ModA')
    plt.bar(x + bar_width/2, mod_means['ModB'], width=bar_width, label='ModB')

    plt.xticks(x, mod_means['Status'])
    plt.ylabel('Average Value')
    plt.title('Average ModA & ModB: Pre-clean vs Post-clean')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_corr_heatmap(df, cols, title="Correlation Heatmap"):
    """
    Plots a heatmap of correlations for the specified columns.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        cols (list): List of column names to include in the correlation.
        title (str): Title for the plot.
    """

    corr_matrix = df[cols].corr()
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_scatter_vs_base(df, cols, base_col, alpha=0.5):
    """
    Plots scatter plots of each column in cols vs. base_col.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        cols (list): List of columns to plot on the y-axis.
        base_col (str): The column to plot on the x-axis.
        alpha (float): Transparency for scatter points.
    """

    for col in cols:
        plt.figure(figsize=(6, 4))
        plt.scatter(df[base_col], df[col], alpha=alpha)
        plt.xlabel(base_col)
        plt.ylabel(col)
        plt.title(f'{col} vs. {base_col}')
        plt.tight_layout()
        plt.show()


def plot_wind_rose(df, ws_col='WS', wd_col='WD', bins=8):
    """
    Plots a simple wind rose (radial bar plot) using matplotlib only.

    Parameters:
        df (pd.DataFrame): DataFrame with wind speed and direction.
        ws_col (str): Name of wind speed column.
        wd_col (str): Name of wind direction column (degrees).
        bins (int): Number of direction bins (e.g., 8 for N, NE, E, ...).
    """

    # Bin wind directions
    wd = df[wd_col] % 360
    ws = df[ws_col]
    wd_bins = np.linspace(0, 360, bins+1)
    wd_centers = (wd_bins[:-1] + wd_bins[1:]) / 2

    # Calculate mean wind speed per direction bin
    ws_means = []
    for i in range(bins):
        mask = (wd >= wd_bins[i]) & (wd < wd_bins[i+1])
        ws_means.append(ws[mask].mean() if mask.any() else 0)

    # Plot
    angles = np.deg2rad(wd_centers)
    ws_means.append(ws_means[0])  # close the circle
    angles = np.append(angles, angles[0])

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.bar(angles, ws_means, width=2*np.pi/bins, bottom=0.0, color='skyblue', edgecolor='k', alpha=0.7)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad(np.linspace(0, 360, bins, endpoint=False)))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][:bins])
    plt.title('Wind Rose (WS/WD)')
    plt.show()

def plot_histograms(df, cols, bins=30, kde=True, colors=None):
    """
    Plots histograms for the specified columns side by side.

    Parameters:
        df (pd.DataFrame): DataFrame with data.
        cols (list): List of column names to plot.
        bins (int): Number of bins for the histogram.
        kde (bool): Whether to plot kernel density estimate.
        colors (list): List of colors for each histogram.
    """

    n = len(cols)
    if colors is None:
        colors = sns.color_palette("husl", n)

    plt.figure(figsize=(6*n, 5))
    for i, col in enumerate(cols):
        plt.subplot(1, n, i+1)
        sns.histplot(df[col], bins=bins, kde=kde, color=colors[i])
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
