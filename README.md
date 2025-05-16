# Stacking Regressor for Regression

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

This repository implements a **stacking regressor** for regression tasks, combining multiple machine learning models (`XGBRegressor`, `RandomForestRegressor`) with a meta-regressor (`RidgeCV`) to predict continuous target variables. The project includes data preprocessing, model training, 5-fold cross-validation, and evaluation, achieving a **mean CV RMSE of 13.0075** on a sample dataset.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)



## Project Overview

The stacking regressor leverages ensemble learning to improve prediction accuracy by combining the strengths of multiple base models. Key components include:

- **Base Estimators**: `XGBRegressor` (gradient boosting) and `RandomForestRegressor` (tree-based ensemble).
- **Meta-Regressor**: `RidgeCV` to combine predictions from base models.
- **Cross-Validation**: 5-fold CV to assess model performance.
- **Evaluation Metrics**: Root Mean Squared Error (RMSE) and R² score.

The project is designed for tabular regression datasets and includes preprocessing (scaling, splitting) and error handling for common issues, such as the `XGBRegressor` early stopping error. The example uses the [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) but can be adapted to any regression task.

## Features

- **Stacking Ensemble**: Combines `XGBRegressor` and `RandomForestRegressor` for robust predictions.
- **Cross-Validation**: Reports mean CV RMSE (13.0075) using 5-fold CV.
- **Preprocessing**: Standardizes features and splits data into train/validation/test sets.
- **Error Handling**: Avoids `XGBRegressor` early stopping issues by disabling `early_stopping_rounds`.
- **Extensibility**: Easily add new estimators or adapt to custom datasets.
- **Evaluation**: Computes RMSE and R² on validation and test sets.

## Requirements

- **Python**: 3.8 or higher
- **Dependencies**:
  - `numpy>=1.21.0`
  - `pandas>=1.3.0`
  - `scikit-learn>=1.0.0`
  - `xgboost>=1.7.0`

See [`requirements.txt`](#code-structure) for exact versions.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/stacking-regressor.git
   cd stacking-regressor
