import pytest
import pandas as pd
import numpy as np
from profiling_and_cleaning import (
    plot_outlier_boxplots, add_outlier_flags,
    report_na_and_duplicates, clean_na_and_duplicates
)

@pytest.fixture
def sample_df():
    df = pd.DataFrame({
        'A': [1, 2, 2, np.nan, 5, 100],
        'B': [10, 20, 20, 30, np.nan, 200],
        'C': [5, 5, 5, 5, 5, 5]
    })
    return df

def test_plot_outlier_boxplots(sample_df):
    # Should run without error
    plot_outlier_boxplots(sample_df.fillna(0), ['A', 'B', 'C'])

def test_add_outlier_flags(sample_df):
    df_flagged = add_outlier_flags(sample_df.fillna(0), ['A', 'B'])
    assert 'outlier_A' in df_flagged.columns
    assert 'outlier_B' in df_flagged.columns
    assert df_flagged['outlier_A'].dtype == bool or df_flagged['outlier_A'].dtype == np.bool_

def test_report_na_and_duplicates(capsys, sample_df):
    report_na_and_duplicates(sample_df)
    captured = capsys.readouterr()
    assert "Missing Values Report:" in captured.out
    assert "Number of duplicated rows:" in captured.out

def test_clean_na_and_duplicates(sample_df):
    cleaned = clean_na_and_duplicates(sample_df)
    # Should remove rows with NA and duplicates
    assert cleaned.isna().sum().sum() == 0
    assert cleaned.duplicated().sum() == 0