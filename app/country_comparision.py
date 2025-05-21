
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import f_oneway, kruskal

def plot_country_boxplots(df, metrics, country_col='Country', palette=None, figsize=(10, 5)):
    """
    Plots boxplots for specified metrics grouped by country.

    Parameters:
        df (pd.DataFrame): Combined DataFrame with a country column.
        metrics (list): List of metric column names to plot.
        country_col (str): Name of the country column.
        palette (dict or None): Color palette for countries.
        figsize (tuple): Figure size for each plot.
    """

    if palette is None:
        unique_countries = df[country_col].unique()
        palette = dict(zip(unique_countries, sns.color_palette("tab10", len(unique_countries))))

    for metric in metrics:
        plt.figure(figsize=figsize)
        sns.boxplot(
            data=df,
            x=country_col,
            y=metric,
            hue=country_col,
            palette=palette,
            legend=False
        )
        plt.xlabel(country_col)
        plt.ylabel(metric)
        plt.title(f"Boxplot of {metric} by {country_col}")
        plt.show()

def print_country_metric_extremes(summary, metrics=['GHI', 'DNI', 'DHI'], stats=['mean', 'median', 'std']):
    """
    Prints the country with the highest and lowest value for each metric and statistic.

    Parameters:
        summary (pd.DataFrame): MultiIndex DataFrame with metrics and stats (as from groupby().agg()).
        metrics (list): List of metric names (columns).
        stats (list): List of statistics to check (e.g., mean, median, std).
    """
    for metric in metrics:
        print(f"\n=== {metric} ===")
        for stat in stats:
            values = summary[(metric, stat)]
            max_country = values.idxmax()
            min_country = values.idxmin()
            max_value = values.max()
            min_value = values.min()
            print(f"{stat.capitalize()}:")
            print(f"  Highest: {max_country} ({max_value:.2f})")
            print(f"  Lowest:  {min_country} ({min_value:.2f})")

def run_anova(*groups):
    """
    Runs a one-way ANOVA test on the provided groups.
    Args:
        *groups: Variable number of arrays/lists/Series to compare.
    Returns:
        p-value (float)
    """
    # Remove NaNs from each group
    cleaned = [pd.Series(g).dropna() for g in groups]
    stat, p = f_oneway(*cleaned)
    return p

def run_kruskal(*groups):
    """
    Runs a Kruskal–Wallis H-test on the provided groups.
    Args:
        *groups: Variable number of arrays/lists/Series to compare.
    Returns:
        p-value (float)
    """
    cleaned = [pd.Series(g).dropna() for g in groups]
    stat, p = kruskal(*cleaned)
    return p

def plot_avg_ghi_bar(df, country_col='Country', ghi_col='GHI', hue= "Countyr", figsize=(6,4)):
    """
    Plots a bar chart ranking countries by average GHI.

    Parameters:
        df (pd.DataFrame): DataFrame containing country and GHI columns.
        country_col (str): Name of the country column.
        ghi_col (str): Name of the GHI column.
        figsize (tuple): Figure size.
    """
    avg_ghi = df.groupby(country_col)[ghi_col].mean().sort_values(ascending=False)
    plt.figure(figsize=figsize)
    sns.barplot(x=avg_ghi.index, y=avg_ghi.values, palette="tab10")
    plt.ylabel('Average GHI')
    plt.xlabel('Country')
    plt.title('Average GHI by Country')
    plt.tight_layout()
    plt.show()