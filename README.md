# House Price Analysis

A complete data analysis project for predicting California house prices, built with Python, scikit-learn, pandas, and matplotlib.

## Project Overview

This project performs end-to-end data analysis and machine learning on a California housing dataset (20,640 samples), covering:

- Exploratory Data Analysis (EDA)
- Data cleaning & feature engineering
- Multiple regression model comparison
- Geographic visualization
- Model performance evaluation

## Project Structure

```
house-price-analysis/
├── data/
│   ├── raw/                        # Raw data (auto-generated on first run)
│   │   └── california_housing.csv
│   └── processed/                  # Cleaned data
│       └── cleaned_data.csv
├── notebooks/
│   └── analysis.ipynb              # Interactive analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Data loading module
│   ├── preprocessing.py            # Cleaning & feature engineering
│   └── visualization.py            # All chart generation
├── outputs/
│   ├── figures/                    # Generated charts (7 PNG files)
│   └── reports/
│       └── model_performance.txt   # Model comparison report
├── main.py                         # Main entry point
├── requirements.txt
├── .gitignore
└── README.md
```

## Features

| Feature | Description |
|---------|-------------|
| MedInc | Median income in block group ($10K) |
| HouseAge | Median house age (years) |
| AveRooms | Average number of rooms |
| AveBedrms | Average number of bedrooms |
| Population | Block group population |
| AveOccup | Average household occupancy |
| Latitude | Block group latitude |
| Longitude | Block group longitude |

**Target**: `MedHouseVal` — Median house value (×$100K)

## Outputs

| File | Description |
|------|-------------|
| `01_target_distribution.png` | House price distribution (raw + log-transformed) |
| `02_correlation_heatmap.png` | Feature correlation heatmap |
| `03_feature_vs_target.png` | Key features vs price scatter plots |
| `04_geographic_distribution.png` | Geographic price heat map |
| `05_model_comparison.png` | RMSE & R² comparison across models |
| `06_prediction_vs_actual.png` | Best model: predicted vs actual |
| `07_feature_importance.png` | Feature importance (Gradient Boosting) |

## Model Results

| Model | RMSE | R² |
|-------|------|-----|
| Linear Regression | 0.4669 | 0.8353 |
| Ridge | 0.4669 | 0.8353 |
| Random Forest | 0.4788 | 0.8268 |
| **Gradient Boosting** | **0.4664** | **0.8357** |

> Best model: **Gradient Boosting** with R² = 0.8357

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/house-price-analysis.git
cd house-price-analysis

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run full pipeline
python main.py
```

Results will be saved to `outputs/figures/` and `outputs/reports/`.

## Tech Stack

- **Python 3.8+**
- **pandas / numpy** — data manipulation
- **scikit-learn** — preprocessing, modeling, evaluation
- **matplotlib / seaborn** — visualization

## Key Findings

1. **Income (MedInc)** is the strongest predictor of house price (correlation > 0.7).
2. **Geographic location** (latitude/longitude) shows clear price clusters — coastal areas command premium prices.
3. **Gradient Boosting** achieves the best performance with R² = 0.8357, outperforming linear models in capturing non-linear relationships.
4. **Feature engineering** (RoomsPerPerson, BedroomRatio, IncomeAge) contributed to improved model performance.

## License

MIT License
