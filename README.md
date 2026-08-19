# EC-KNN-ST + Efficiency/Accuracy Modification — Code

This repository contains the implementation of the **EC-KNN-ST** spatiotemporal data imputation algorithm, including an optimized variant designed to maintain the accuracy of the complex ensemble version while significantly reducing computational cost through density-gated search and adaptive parameters.

## Files
- `ec_knn_st.py`: Implements three variants of the algorithm:
  - `knn_st_predict`: The baseline algorithm.
  - `ec_knn_st_v2_predict`: The full, computationally expensive ensemble version (evaluates all candidates).
  - `ec_knn_st_modified_predict`: The proposed optimized version (uses density-gated candidate search, two-stage pruning, and adaptive sigma/lambda).
- `make_demo_dataset.py`: Handles data loading and processing. It reads real air pollutant measurements, normalizes the spatial/temporal coordinates, and also contains a fallback synthetic data generator for testing.
- `run_experiment.py`: The main entry point. It evaluates the three methods at missing rates from 10% to 90%, reporting Normalized Error (NE) and candidate-evaluation counts (as a proxy for computational cost).
- `Seoul_dataset.csv`: The real dataset containing PM10, PM2.5, and CO measurements from Seoul stations. 

## How to Run the Experiments

The codebase is already configured to run with the real `Seoul_dataset.csv`.

Simply execute the experiment script:
```bash
python run_experiment.py
```

**Note on rigorous evaluation:**
Currently, `run_experiment.py` evaluates using 5 random masks per setting (`n_repeats=5`). For a fully rigorous comparison against existing v1/v2 benchmarks (as per Section 5.4 of the report), you should change `n_repeats=5` to `n_repeats=20` on line 69 of `run_experiment.py`.

## What's real vs. synthetic in this repo
- **REAL**: The repository natively uses the real data from `Seoul_dataset.csv` out of the box. The data-loading logic in `make_demo_dataset.py` successfully normalizes and feeds this data into the experiment pipeline.
- **SYNTHETIC**: A synthetic generator (`synthetic_pollutant_field`) remains available in `make_demo_dataset.py` as a lightweight fallback or for quick structural testing without loading the full dataset.
