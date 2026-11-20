The full runnable code is in [Default_loan_prediction_csv_updated.ipynb](Default_loan_prediction_csv_updated.ipynb)
# Loan Default Prediction | Python, Data Science & Machine Learning
Predicts whether a borrower will default on a loan from employment status, bank balance and annual salary. The project shows how an imbalanced target (only 3.3% defaults) makes accuracy misleading, and how class weighting and better metrics give a more honest evaluation.

## Dataset

- 10,000 borrower records, no missing values, no duplicates
- Features: `Employed`, `Bank Balance`, `Annual Salary` (the `Index` column is dropped)
- Target: `Defaulted?` (1 = default, 0 = no default)
- Strong class imbalance: 333 defaults (3.3%) vs 9,667 non-defaults (96.7%)
- Train/test split: 70% / 30%

## Results

### Step 1: Baseline (plain Logistic Regression)

| Metric | Value |
|---|---|
| Accuracy | 97.3% |
| Precision (defaulters) | 0.69 |
| Recall (defaulters) | 0.27 |
| F1-score (defaulters) | 0.38 |

|  | Predicted: No default | Predicted: Default |
|---|---|---|
| **Actual: No default** | 2,895 | 11 |
| **Actual: Default** | 69 | 25 |

A model that predicts "no default" for everyone scores 96.9% on this test set, so 97.3% accuracy is barely better than guessing. The baseline catches only 27% of real defaulters.

### Step 2: Improved models (stratified split, scaled features, class weights)

Baseline accuracy (always predict "no default"): 96.7%. Test set: 3,000 records, 100 defaulters.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.863 | 0.182 | 0.890 | 0.303 | 0.951 |
| Decision Tree | 0.849 | 0.159 | 0.820 | 0.266 | 0.885 |
| Random Forest | 0.967 | 0.517 | 0.310 | 0.388 | 0.891 |

**Best at ranking risk:** Logistic Regression (ROC-AUC 0.951). **Highest F1:** Random Forest (0.39), but it catches only 31 of 100 defaulters.

**Trade-off:** class-balanced Logistic Regression catches 89 of 100 defaulters (recall 0.27 → 0.89) but flags 399 non-defaulters by mistake (precision 0.18). Random Forest raises far fewer false alarms (29) but misses 69% of defaulters. Which is better depends on cost: if missing a defaulter is far more expensive than reviewing a false alarm, the high-recall model wins.

**Top features (Random Forest):** Bank Balance (0.80), Annual Salary (0.19), Employed (0.01). Bank balance is by far the strongest signal.

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Models:** Logistic Regression, Decision Tree, Random Forest

## Workflow

1. Data loading and cleaning (missing values, duplicates)
2. Exploratory data analysis (class balance, income distribution)
3. Feature selection (drop the `Index` column)
4. Stratified train/test split and feature scaling
5. Baseline model, then class-balanced models
6. Evaluation: confusion matrix, precision, recall, F1, ROC-AUC

## How to Run

```bash
git clone https://github.com/sankalplohe-ai/loan-default-prediction.git
cd loan-default-prediction
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook
```

Place `Default_loan_prediction.csv` in the same folder as the notebook.

## Conclusion

Accuracy alone overstated the baseline model on this imbalanced data. Class weighting and evaluating precision, recall, F1 and ROC-AUC on the defaulter class gave a more honest picture and a clear precision/recall trade-off. Possible next steps: decision-threshold tuning, resampling (SMOTE) and a cost-sensitive evaluation.

## Author
Sankalpa Ratnabodhi Lohe · sankalplohe@gmail.com · [LinkedIn](https://www.linkedin.com/in/sankalpa-mahendra-ratnabodhi-lohe-88098b200)
