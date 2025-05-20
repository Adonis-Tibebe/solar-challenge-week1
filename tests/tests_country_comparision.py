import pytest
import pandas as pd
import numpy as np

from country_comparision import (
    plot_country_boxplots,
    print_country_metric_extremes,
    run_anova,
    run_kruskal,
    plot_avg_ghi_bar
)

@pytest.fixture
def sample_df():
    data = {
        'Country': ['A', 'A', 'B', 'B', 'C', 'C'],
        'GHI': [10, 12, 20, 22, 30, 32],
        'DNI': [5, 6, 7, 8, 9, 10],
        'DHI': [1, 2, 3, 4, 5, 6]
    }
    return pd.DataFrame(data)

@pytest.fixture
def summary_df(sample_df):
    return sample_df.groupby('Country')[['GHI', 'DNI', 'DHI']].agg(['mean', 'median', 'std'])

def test_run_anova(sample_df):
    # Should return a float p-value
    p = run_anova(sample_df[sample_df['Country'] == 'A']['GHI'],
                  sample_df[sample_df['Country'] == 'B']['GHI'],
                  sample_df[sample_df['Country'] == 'C']['GHI'])
    assert isinstance(p, float)
    assert 0 <= p <= 1

def test_run_kruskal(sample_df):
    p = run_kruskal(sample_df[sample_df['Country'] == 'A']['GHI'],
                    sample_df[sample_df['Country'] == 'B']['GHI'],
                    sample_df[sample_df['Country'] == 'C']['GHI'])
    assert isinstance(p, float)
    assert 0 <= p <= 1

def test_print_country_metric_extremes(summary_df, capsys):
    print_country_metric_extremes(summary_df, metrics=['GHI', 'DNI', 'DHI'], stats=['mean', 'median', 'std'])
    captured = capsys.readouterr()
    assert "Highest" in captured.out
    assert "Lowest" in captured.out

def test_plot_country_boxplots(sample_df):
    # Should not raise errors
    plot_country_boxplots(sample_df, ['GHI', 'DNI', 'DHI'])

def test_plot_avg_ghi_bar(sample_df):
    # Should not raise errors
    plot_avg_ghi_bar(sample_df)