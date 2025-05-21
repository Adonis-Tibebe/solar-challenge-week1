import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

def plot_outlier_boxplots(df, cols, figsize=(15, 8)):
    """
    Plots boxplots for the specified columns to visualize outliers.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        cols (list): List of column names to plot.
        figsize (tuple): Figure size for the plot.
    """
    

    plt.figure(figsize=figsize)
    sns.boxplot(data=df[cols])
    plt.title('Boxplots for Outlier Detection')
    plt.tight_layout()
    plt.show()

def add_outlier_flags(df, cols, threshold=3):
    """
    Adds boolean outlier flag columns to the DataFrame using Z-score method.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        cols (list): List of column names to check for outliers.
        threshold (float): Z-score threshold for flagging outliers.
    Returns:
        pd.DataFrame: DataFrame with new outlier flag columns.
    """

    z_scores = np.abs(stats.zscore(df[cols], nan_policy='omit'))
    for idx, col in enumerate(cols):
        df[f'outlier_{col}'] = z_scores[:, idx] > threshold
    return df

def report_na_and_duplicates(df):
    """
    Prints a summary of NA (missing) values and duplicate rows in the DataFrame.
    """
    na_counts = df.isna().sum()
    na_percent = (df.isna().mean() * 100).round(2)
    duplicates = df.duplicated().sum()
    print("Missing Values Report:")
    print(pd.DataFrame({'count': na_counts, 'percent': na_percent})[na_counts > 0])
    print(f"\nNumber of duplicated rows: {duplicates}")

def clean_na_and_duplicates(df, how_na='any', subset_na=None, subset_dup=None):
    """
    Removes NA values and duplicate rows from the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        how_na (str): 'any' or 'all' for dropping NA (default 'any').
        subset_na (list): Columns to consider for NA removal (default None = all columns).
        subset_dup (list): Columns to consider for duplicate removal (default None = all columns).
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.dropna(how=how_na, subset=subset_na)
    df_clean = df_clean.drop_duplicates(subset=subset_dup)
    return df_clean