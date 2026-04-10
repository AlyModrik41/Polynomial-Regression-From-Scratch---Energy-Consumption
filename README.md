# Polynomial Regression — Energy Consumption Prediction

Polynomial regression implemented from scratch using gradient descent, with Ridge and Lasso regularization via sklearn. Built as part of an AI course assignment.

## Dataset
Residential energy consumption dataset with 4 features:
- **Square Footage** — strongest predictor (correlation: 0.77)
- **Number of Occupants** — moderate correlation (0.35)
- **Appliances Used** — moderate correlation (0.31)
- ~~Average Temperature~~ — dropped (correlation: -0.034)

## What It Does
- Builds cumulative polynomial features up to a specified degree (including interaction terms)
- Trains a plain linear regression model using gradient descent
- Compares against Ridge (α=150) and Lasso (α=150) regularization
- Plots Train MSE, Test MSE, Ridge MSE, and Lasso MSE across degrees 1–9

## Results

| Degree | Train MSE | Test MSE | Ridge MSE | Lasso MSE |
|--------|-----------|----------|-----------|-----------|
| 1 | 166,767 | 170,679 | 184,784 | 208,916 |
| 3 | 164,357 | 174,818 | 190,979 | 239,506 |
| 6 | 152,839 | 195,267 | 190,540 | 263,633 |
| 9 | 125,873 | 426,771 | 212,811 | 277,997 |

Degree 1 gives the best test MSE — the data has a predominantly linear structure so higher degrees overfit.

## Setup

```bash
pip install pandas numpy matplotlib scikit-learn seaborn
python plr.py
```

## Key Design Note
Polynomial features are **cumulative** — degree `d` includes all terms from degrees 1 through `d`, plus pairwise interaction terms. An earlier version that only included terms of exactly degree `d` caused erratic oscillating MSE and was corrected.
