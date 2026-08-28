<div align="center">
  <img src="logo.png" alt="Logo" width="128" height="128">
  
  <h1>AXA Financial Indonesia Insurance Claim Forecasting</h1>
  <p><strong>Hierarchical Ensemble Forecasting & Actuarial Risk Analysis for MCF ITB 2026</strong></p>
  
  <p align="center">
    <img src="https://img.shields.io/badge/Competition-MCF_ITB_2026-blue?style=flat-square" alt="Competition">
    <img src="https://img.shields.io/badge/Language-Python_3-3776AB?style=flat-square&logo=python&logoColor=white" alt="Language">
    <img src="https://img.shields.io/badge/Status-Completed-success?style=flat-square" alt="Status">
    <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License">
  </p>
  
  <p align="center">
    An end-to-end time series and machine learning pipeline to forecast monthly insurance claim frequency, claim severity, and total claims for a portfolio of 4,096 policyholders across a 17-month horizon (August 2025 - December 2026).
  </p>
</div>

## Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

## Problem Overview

| Parameter | Detail |
|---|---|
| Historical Data | 19 months (January 2024 - July 2025), 4,627 PAID claims |
| Forecast Horizon | 17 months (August 2025 - December 2026) |
| Target Variables | `Claim_Frequency`, `Claim_Severity`, `Total_Claim` |
| Evaluation Metric | MAPE (Mean Absolute Percentage Error) |
| Core Challenge | Historical-to-forecast ratio is nearly 1:1, posing a high risk of long-range variance and compounding errors |

## Methodology and Architecture

The forecasting system combines three complementary models using inverse MAPE weighting derived from walk-forward cross-validation:

1. **Prophet**: Captures macroeconomic trends, annual progression, and quarterly seasonal oscillations.
2. **LightGBM**: Models non-linear interactions, lag structures (Lags 1-3), rolling statistics (Rolling 3, Rolling 6, Rolling Std), and seasonal trigonometric signals.
3. **Seasonal Naive**: Acts as an empirical statistical baseline anchored to month-of-year historical patterns.

### Inverse MAPE Blending

Base ensemble weights are computed from out-of-fold validation errors:

$$
w_m = rac{rac{1}{\text{MAPE}_m}}{\sum_{k=1}^{K} rac{1}{\text{MAPE}_k}}
$$

### Progressive Weighting for Long Horizon

To prevent ML divergence and error compounding across the 17-month horizon, a progressive weighting mechanism dynamically shifts weight toward the Seasonal Naive baseline over time. By 2026 (the naive zone), Seasonal Naive accounts for 50% - 80% of the blended prediction.

## Notebook Pipeline Structure

The entire research and modeling workflow is implemented in `119_DSC_Cungpret_Notebook.ipynb`:

| Cell | Stage | Description |
|---|---|---|
| Cell 0 | Environment | Library installation and dependencies import |
| Cell 1 | Data Preprocessing | Ingestion, data cleaning, date standardization, and 1%-99% winsorization |
| Cell 2 | Exploratory Data Analysis | Distribution analysis, risk factor profiling, and statistical hypothesis tests |
| Cell 3 | Feature Engineering | Monthly aggregation, exposure projection, lag creation, and cyclical encodings |
| Cell 4 | Validation Strategy | Walk-forward cross-validation and inverse MAPE ensemble weight optimization |
| Cell 4A | Sensitivity Analysis | Prophet hyperparameter grid exploration and naive blending robustness checks |
| Cell 5 | Final Training | Full-dataset model fitting across frequency, severity, and total claim targets |
| Cell 6 | Multi-Step Forecasting | 17-month recursive and direct forecasting with progressive weighting |
| Cell 6A | Uncertainty Estimation | Block bootstrap prediction intervals (80% and 90% confidence bands) |
| Cell 6B | Empirical Backtesting | 9-window historical coverage backtesting for prediction intervals |
| Cell 7 | Submission Generation | Formatting and exporting competition predictions (`submission.csv`) |
| Cell 8 | Feature Ablation Study | Quantitative evaluation of feature contributions against baseline models |
| Cell 9A | Micro-Level Analysis | Claim-level driver identification using Kruskal-Wallis and Spearman correlation |
| Cell 9B | Macro-Level Analysis | Portfolio-level LightGBM feature importance and SHAP interpretation |
| Cell 10 | Artifact Export | Serializing trained models and metadata to `models/` |
| Cell 11 | Strategic Insights | Actuarial risk recommendations and business takeaways |

## Data Preprocessing and Feature Engineering

### Data Sources

- `Data_Polis.csv`: Contains 4,096 active policyholders with demographic information (Gender, Date of Birth, Policy Effective Date) and insurance plans (Plan Code: M-001, M-002, M-003).
- `Data_Klaim.csv`: Contains 5,781 claim transactions with ICD diagnosis codes, service types (Inpatient vs Outpatient), payment methods (Cashless vs Reimbursement), hospital admission/discharge dates, and approved claim amounts.

### Data Cleaning and Transformation

1. Standardized mixed date formats between policy records (`YYYYMMDD`) and claims records.
2. Filtered for approved claims with status `PAID` (4,627 records retained).
3. Resolved duplicate entries based on unique `Claim ID`.
4. Applied 1%-99% two-sided winsorization on claim amounts to protect models against extreme tail outliers while preserving the natural Rupiah currency scale.

Winsorization thresholds:
- Lower Bound: IDR 187,242
- Upper Bound: IDR 633,686,130

### Engineered Features

| Feature | Type | Formulation / Description |
|---|---|---|
| `Age` | Demographic | Calculated relative to reference date (July 1, 2025) |
| `Policy_Tenure` | Portfolio | Elapsed years since policy effective date |
| `Age_Group` | Categorical | Five actuarial brackets: <18, 18-30, 31-45, 46-60, >60 |
| `Length_of_Stay` | Utilization | Days between hospital admission and discharge |
| `Claim_Ratio` | Financial | Approved claim amount divided by total hospital billed charges |
| `Trigonometric Cyclical` | Temporal | Sine and cosine encodings for month and quarter cycles |
| `Lags & Rolling Windows` | Time Series | Lag 1-3, Rolling Mean 3, Rolling Mean 6, and Rolling Standard Deviation |

## Results and Performance

### Validation Metrics

| Metric | Result |
|---|---|
| Internal Walk-Forward Validation MAPE | 4.79% |
| Public Leaderboard MAPE (First 5 Months) | 5.80% |
| Total Claim 80% Prediction Interval (2026) | IDR 132.34 - 149.11 Billion |
| Empirical Coverage (9-Window Backtesting) | 88.9% for Total Claim |

### 2026 Annual Portfolio Forecast

| Target Variable | 2026 Point Forecast | Annual Trend vs 2025 Annualized |
|---|---|---|
| Total Claim Frequency | ~2,827 claims | +0.8% |
| Average Claim Severity | ~IDR 49.40 Million / claim | -3.0% |
| Total Claim (Point Estimate) | IDR 143.00 Billion | Stable growth trajectory |
| Total Claim (P50 Bootstrap) | IDR 143.49 Billion | High median consistency |

### 2026 Quarterly Technical Reserve Allocations

| Quarter | Point Estimate | Conservative P90 Reserve |
|---|---|---|
| Q1 2026 (Jan - Mar) | IDR 37.13 Billion | IDR 38.70 Billion |
| Q2 2026 (Apr - Jun) | IDR 33.32 Billion | IDR 34.74 Billion |
| Q3 2026 (Jul - Sep) | IDR 36.01 Billion | IDR 37.54 Billion |
| Q4 2026 (Oct - Dec) | IDR 36.55 Billion | IDR 38.13 Billion |
| **Full Year 2026 Total** | **IDR 143.00 Billion** | **IDR 149.11 Billion** |

## Key Findings and Exploratory Insights

- **Severity Distribution**: Heavily right-skewed; over 80% of claims are under IDR 50 Million, with the main concentration below IDR 20 Million.
- **Age Concentration**: Policyholders aged >60 generate over 2,200 claims with the highest median claim cost across all cohorts.
- **Diagnosis Patterns**: ICD code `N18.0` (End-stage renal disease) represents the single largest volume of recurring claims (~300 claims), followed by `C50` (Breast malignant neoplasm) and `H26` (Cataract).
- **Hospitalization Dynamics**: `Length_of_Stay` exhibits the highest numerical correlation with claim amount (Spearman rho = 0.503, p < 0.001). Inpatient claims have a 9.94x higher median cost than Outpatient claims.
- **Payment Method Effect**: Cashless claims consistently yield higher median payouts (~IDR 20 Million) compared to Reimbursement claims (<IDR 10 Million).
- **Seasonality**: Historical claim volume peaks annually in January (302 claims, IDR 18.95 Billion in Jan 2024), followed by stable monthly activity between 208 and 278 claims.

## Exported Model Artifacts

All trained models and ensemble metadata are stored in the `models/` directory:

| Artifact File | Model Architecture | Target Description |
|---|---|---|
| `lgbm_freq.pkl` | LightGBM Regressor | Monthly Claim Frequency |
| `lgbm_sev.pkl` | LightGBM Regressor | Monthly Average Claim Severity |
| `lgbm_total.pkl` | LightGBM Regressor | Monthly Total Claim Amount |
| `prophet_freq.pkl` | Facebook Prophet | Claim Frequency Trend & Seasonality |
| `prophet_sev.pkl` | Facebook Prophet | Claim Severity Trend & Seasonality |
| `prophet_total.pkl` | Facebook Prophet | Total Claim Trend & Seasonality |
| `seasonal_naive.json` | Statistical Baseline | Empirical monthly lookup values |
| `ensemble_meta.json` | Configuration | Ensemble weights and feature specifications |

## Repository Structure

```
.
├── 119_DSC_Cungpret_Notebook.ipynb    # Main pipeline notebook (Cells 0 - 11)
├── Data_Polis.csv                    # Policyholder portfolio dataset
├── Data_Klaim.csv                    # Historical claims dataset
├── submission.csv                    # Final generated forecast submission
├── sample_submission.csv             # Kaggle submission template
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore patterns
├── LICENSE                           # MIT License
├── logo.png                          # Repository logo
├── case_study/                       # Competition problem, guidebook, & report
│   ├── 119_DSC_Cungpret_Laporan.pdf
│   ├── case_study.pdf
│   └── guidebook.pdf
├── certificates/                     # Competition finalist certificates
│   ├── certificate_ababil_khoerul_imam.jpg
│   ├── certificate_ferliyana_ronnan.jpg
│   └── certificate_vierico_ventora.jpg
└── models/                           # Serialized model binaries and metadata
    ├── ensemble_meta.json
    ├── seasonal_naive.json
    ├── lgbm_freq.pkl
    ├── lgbm_sev.pkl
    ├── lgbm_total.pkl
    ├── prophet_freq.pkl
    ├── prophet_sev.pkl
    └── prophet_total.pkl
```

## Getting Started

### Prerequisites

Ensure Python 3.9+ is installed in your environment.

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/ababilkhoerulimam/insurance-claim-forecasting-mcfitb26.git
cd insurance-claim-forecasting-mcfitb26
pip install -r requirements.txt
```

### Execution

Open and run `119_DSC_Cungpret_Notebook.ipynb` sequentially from Cell 0 through Cell 11. Ensure that `Data_Polis.csv` and `Data_Klaim.csv` are in the project root directory.

## Methodological Notes

- The Total Claim prediction interval achieves an empirical coverage of 88.9% across 9 backtesting windows, surpassing the 80% nominal target.
- Claim Frequency and Claim Severity prediction intervals provide indicative bands with an empirical coverage of 66.7%.
- The 2026 forecast applies a progressive blend giving 50% - 80% weight to the Seasonal Naive baseline to safeguard against model extrapolation drift given the 19-month historical sample size.

## Team Cungpret

MCF ITB 2026 Finalists:

- Ababil Khoerul Imam ([Certificate](certificates/certificate_ababil_khoerul_imam.jpg))
- Ferliyana Ronnan ([Certificate](certificates/certificate_ferliyana_ronnan.jpg))
- Vierico Ventora ([Certificate](certificates/certificate_vierico_ventora.jpg))

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

