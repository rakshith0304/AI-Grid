# AI-Grid

AI-Grid is a capstone research project for exploring U.S. electricity generation patterns and
classifying monthly state-level grid behavior as relatively "green" or "dirty" based on
renewable share and related features.

The project combines:
- exploratory analysis notebooks,
- a shared data-preparation pipeline,
- a transformer-based classifier for monthly sequences,
- and utility functions for evaluating renewable trends and pricing relationships.

## Project Goals

- Build reproducible state-month features from EIA and SEDS data.
- Label state-month observations using renewable-share thresholds.
- Train and evaluate a sequence model (`GreenMonthTransformer`) on these labels.
- Analyze renewable-share dynamics, top-performing states, and retail-price correlations.

## Repository Structure

- `nn.ipynb` - Main modeling and analysis notebook.
- `Transformer/nn.ipynb` - Alternate notebook workflow using the same shared modules.
- `AI_Grid_Analysis.ipynb` - Supplemental analysis notebook.
- `shared/data_pipeline.py` - Canonical data loading, cleaning, feature engineering, labeling, and loader construction.
- `shared/model_defs.py` - Reusable model definitions (currently `GreenMonthTransformer`).
- `shared/train_utils.py` - Shared training/evaluation loop helpers.
- `shared/paths_config.py` - Centralized file path configuration and naming aliases.
- `green_dirty_month_data.py` - Backward-compatible wrapper re-exporting `shared.data_pipeline`.
- `Transformer/green_dirty_month_data.py` - Backward-compatible wrapper for notebook compatibility.
- `data/` - Local project data artifacts.

## Data Sources and Inputs

The shared pipeline expects datasets under a `research_data` directory (resolved from
`shared/paths_config.py`), including:
- grid operations monthly power generation extracts,
- SEDS state energy/CO2 data (`TETCE`),
- retail electricity price data,
- ISO/RTO fuel-mix files (CAISO, ERCOT, ISO-NE, NYISO, PJM).

If the preferred monthly power file is missing, the pipeline falls back to the 2010-2024 extract.

## Core Pipeline Overview

`shared/data_pipeline.py` provides:
- State-month power table construction (`load_power_state_month_table`),
- CO2 table loading (`load_state_tetce`),
- Label generation (`build_labeled_frame`),
- Sequence tensor generation (`build_sequence_tensors`),
- PyTorch DataLoader construction (`make_loaders`),
- Analysis helpers for green-state ranking and renewable-price relationships.

## Model and Training

- Model: `GreenMonthTransformer` in `shared/model_defs.py`.
- Training helper: `run_epoch` in `shared/train_utils.py`.
- Notebooks import these shared modules to keep modeling logic centralized and reduce drift.

## Environment

Recommended Python packages:
- `numpy`
- `pandas`
- `scipy`
- `torch`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `jupyter`
