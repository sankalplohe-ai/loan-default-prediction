# Loan Default Prediction | Python, Data Science & Machine Learning

Predicts whether a borrower will default on a loan from employment status, bank balance and annual salary, using a Logistic Regression classifier.

## Dataset

- 10,000 borrower records, no missing values
- Features: Employed, Bank Balance, Annual Salary
- Target: Defaulted? (1 = default, 0 = no default)
- Strong class imbalance: only about 3% of borrowers default
- Train/test split: 70% / 30% (3,000 test records: 2,906 non-defaults, 94 defaults)

## Results (test set)

| Metric | Value |
|---|---|
| Accuracy | 97.3% |
| Precision (defaulters) | 0.69 |
| Recall (defaulters) | 0.27 |
| F1-score (defaulters) | 0.38 |
| ROC-AUC | [__] |

**Confusion matrix**

|  | Predicted: No default | Predicted: Default |
|---|---|---|
| **Actual: No default** | 2,895 | 11 |
| **Actual: Default** | 69 | 25 |

### Key takeaway: accuracy can mislead on imbalanced data

Because only about 3% of borrowers default, a model that predicts "no default" for everyone would already score 96.9% accuracy. This model's 97.3% is only slightly better, and it catches just 27% of actual defaulters. Precision, recall and F1 on the defaulter class are therefore the metrics that matter for this problem.

### Improving recall

[After you rerun with class_weight="balanced", add the new table here, e.g. "Recall on defaulters improved from 0.27 to X, with accuracy changing from 97.3% to Y%."]

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Model:** Logistic Regression

## Workflow

1. Data collection
2. Data cleaning
3. Exploratory data analysis (EDA)
4. Feature selection
5. Model building
6. Prediction and evaluation

## Evaluation Metrics

- Accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score)
- ROC-AUC

## How to Run

```bash
git clone https://github.com/sankalplohe-ai/loan-default-prediction.git
cd loan-default-prediction
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook
```

## Conclusion

The project covers a complete supervised learning workflow, from raw borrower data to an evaluated classifier, and shows why accuracy alone is a poor measure on imbalanced data. Next steps: handle class imbalance (class weights, resampling, threshold tuning) and compare additional models.

## Author

Sankalpa Ratnabodhi · sankalplohe@gmail.com · [LinkedIn](https://www.linkedin.com/in/sankalpamahendra-ratnabodhilohe-88098b200)
