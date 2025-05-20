# Tests

This folder contains automated tests for the project's core modules, using the `pytest` framework.

### What is tested?

- Plotting functions in `plots.py` (to ensure they run without errors)
- Data profiling and cleaning functions in `profiling_and_cleaning.py`
- **Country comparison and statistical analysis functions in `country_comparision.py`** (including boxplots, statistical tests, and summary reporting)

### How to run the tests

From the root or `src` directory, run:

```sh
pytest
```