# CS 6375 - Group 06 - Formula 1 Race Outcome Prediction Using Machine Learning

This project predicts where each driver will finish in a Formula 1 race using only information available before the race starts. Each driver's finishing position is modeled as a regression target, and the predictions within a race are ranked to give a full predicted finishing order.

Features cover the starting grid and qualifying results, driver and constructor championship standings, recent and career performance, circuit history (past results, DNF rates, safety cars, red flags), and race-day weather from the [Open-Meteo](https://open-meteo.com/) archive.

All models are trained and evaluated on chronological splits by season, so they are never trained on races from the future:

| Split | Seasons | Purpose |
| --- | --- | --- |
| Train | 2014-2023 | Model fitting |
| Validation | 2024 | Hyperparameter tuning |
| Test | 2025 | Final evaluation |

We compare the following methods:

- **Baseline:** predicts that every driver finishes where they started on the grid (`notebooks/baseline.ipynb`)
- **Method 1:** Random Forest Regressor (`notebooks/method-01-varun.ipynb`)
- **Method 2:** XGBoost Regressor (`notebooks/method-02-natiq.ipynb`)

Models are scored per race on exact-position accuracy, mean absolute error (MAE) in positions, podium recall, and Spearman rank correlation, then averaged over all races in each split.

## Getting Started

### Prerequisite

Install the latest version of `python` on your system. [Python Download](https://www.python.org/downloads)

### Execute the following commands in a terminal to set up the project

- `git clone https://github.com/heavydriver/f1-race-prediction.git`
- `cd f1-race-prediction`
- `python -m venv .venv`
- `.venv\Scripts\activate.bat` (for Windows) <br /> `source .venv/bin/activate` (for Mac/Linux)
- `pip install -r requirements.txt`
