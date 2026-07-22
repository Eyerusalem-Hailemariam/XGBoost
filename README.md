# XGBoost from Scratch: Customer Churn Prediction

## Overview
XGBoost implementation from scratch using Python and NumPy with exact greedy split-finding algorithm. Benchmarked against official XGBoost on customer churn prediction.

## Dataset: BankChurners (Credit Card Customer Churn)
**Why this dataset?**

1. **Realistic class imbalance** (16.1% churn) - Not a toy 50/50 split. Validates `scale_pos_weight`, threshold tuning, and proper evaluation with precision/recall/F1 instead of misleading accuracy metrics.

2. **Optimal size for exact greedy algorithm** (10,127 rows, ~20 features) - Small enough that exact greedy—which sorts every feature and evaluates gain at every candidate split point with no approximation—remains computationally practical, while large enough to produce non-trivial tree ensembles and meaningful train/val/test splits (5,670/1,418/3,039). This is the sweet spot: exact greedy is XGBoost's default for smaller datasets before switching to histogram-based approximation as data volume grows.

3. **Built-in data quality lesson** - Dataset includes explicit target leakage (`Naive_Bayes_Classifier_*` columns) that must be identified and removed. Mirrors real-world pitfalls where features derived from the target accidentally leak into training.

**Stats**: 10,127 samples | 38 features (post-encoding) | 84% retained, 16% churned

## Key Results

| Metric | Scratch | Official XGBoost | Training Time |
|--------|---------|------------------|---------------|
| **AUC** | 0.9926 | 0.9925 | 35s vs 2.2s (**16× faster**) |
| **F1** | 0.9106 | 0.9055 | Hist method: **27× faster** |
| **Precision** | 0.9241 | 0.8942 | |
| **Recall** | 0.8975 | 0.9180 | |

**Key Insight**: Official XGBoost is much faster (C++ optimized) with identical accuracy. Scratch implementation validates algorithmic correctness.

## Hyperparameters
```python
max_depth=5, learning_rate=0.1, n_estimators=300 (early stopped at 158)
scale_pos_weight=5.22 (handles 16% churn imbalance)
subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0
```

## Top Features (by gain)
1. Total_Trans_Ct, 2. Total_Trans_Amt, 3. Total_Revolving_Bal, 4. Total_Relationship_Count, 5. Contacts_Count_12_mon

## Files
- `XGBoost.py` - Standalone implementation
- `XGBoost_Implementation_labeled.ipynb` - Notebook with labels
- `README.md` - This file
