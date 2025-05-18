import pytest
import pandas as pd
import numpy as np
from plots import (
    plot_rolling_average, plot_hourly_trend, plot_anomalies, plot_mod_cleaning,
    plot_corr_heatmap, plot_scatter_vs_base, plot_wind_rose, plot_histograms
)

@pytest.fixture
def sample_df():
    # Create a small DataFrame for testing
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    df = pd.DataFrame({
        'Timestamp': dates,
        'GHI': np.random.rand(10) * 1000,
        'DNI': np.random.rand(10) * 800,
        'DHI': np.random.rand(10) * 500,
        'Tamb': np.random.rand(10) * 40,
        'ModA': np.random.rand(10) * 10,
        'ModB': np.random.rand(10) * 10,
        'WS': np.random.rand(10) * 5,
        'WSgust': np.random.rand(10) * 7,
        'WD': np.random.rand(10) * 360,
        'RH': np.random.rand(10) * 100,
        'hour': np.arange(10),
        'Cleaning': [0, 1]*5
    })
    return df

def test_plot_rolling_average(sample_df):
    sample_df = sample_df.set_index('Timestamp')
    plot_rolling_average(sample_df, 'GHI')

def test_plot_hourly_trend(sample_df):
    plot_hourly_trend(sample_df, ['GHI', 'DNI'])

def test_plot_anomalies(sample_df):
    sample_df['outlier_GHI'] = [False]*9 + [True]
    plot_anomalies(sample_df, 'GHI', 'outlier_GHI')

def test_plot_mod_cleaning(sample_df):
    plot_mod_cleaning(sample_df)

def test_plot_corr_heatmap(sample_df):
    plot_corr_heatmap(sample_df, ['GHI', 'DNI', 'DHI'])

def test_plot_scatter_vs_base(sample_df):
    plot_scatter_vs_base(sample_df, ['GHI', 'DNI'], 'Tamb')

def test_plot_wind_rose(sample_df):
    plot_wind_rose(sample_df, ws_col='WS', wd_col='WD', bins=8)

def test_plot_histograms(sample_df):
    plot_histograms(sample_df, ['GHI', 'DNI'], bins=5)