# Customer Churn Analysis

An e-commerce company wants to predict which users will churn so it can offer them targeted
promotions. This repository contains the two mini projects built on the same customer dataset.

| Project | Scope | Notebook | Colab |
|---|---|---|---|
| **Mini Project 1 – EDA** | Data understanding, cleaning, descriptive statistics, relationship analysis, business recommendations | [`mini_project_1_eda.ipynb`](mini_project_1_eda.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Thanhnt-da/customer-churn-analysis/blob/main/mini_project_1_eda.ipynb) |
| **Mini Project 2 – Churn Model** | Full ML pipeline: data quality, preprocessing, train/test split, logistic regression and gradient boosting, evaluation, conclusions | [`mini_project_2_model.ipynb`](mini_project_2_model.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Thanhnt-da/customer-churn-analysis/blob/main/mini_project_2_model.ipynb) |

---

## Mini Project 1 – Exploratory Data Analysis

**Question:** what does a churned customer look like, and what can the company do to reduce churn?

### Files
- `mini_project_1_eda.ipynb`: the notebook with all code, charts and observations
- `mini_project_1_eda.py`: the same analysis as a Python script (VS Code "percent" cell format)
- `customer_churn_dataset.csv`: the dataset

### Approach
1. **Data understanding:** size, data types, missing values, duplicates, target distribution
2. **Data cleaning:** removed 1 fully empty row, fixed data types, checked ranges, outliers and category labels
3. **Descriptive statistics:** distributions of the numeric and categorical features
4. **Relationship analysis:** churn rate per feature group, correlation, robustness check across data sources
5. **Insights & recommendations**

### Data-quality finding
The file is **two datasets stacked together** (source A: 440,832 rows, source B: 64,374 rows), and
source B restarts its customer IDs at 1. The 62,995 "duplicate" IDs are therefore different customers
sharing a number, not repeated records. Deleting them would remove 25% of the data, most of it churners
(72% churn against 50% for the rest). Instead, every row was kept and labelled with a `source` column.

### Key findings
Overall churn rate: **55.5%**

| Driver | Churn in group | Churn outside | Holds in both sources? |
|---|---|---|---|
| Support calls ≥ 5 | 90.8% | 35.5% | ✅ |
| Payment delay > 20 days | 94.2% | 44.5% | ✅ |
| Total spend ≤ 500 | 90.2% | 41.4% | ⚠️ weaker in B |
| Monthly contract | 90.2% | 46.0% | ⚠️ weaker in B |
| Age > 50 | 90.2% | 46.6% | ⚠️ weaker in B |
| Female | 64.9% | 48.0% | ✅ |

Subscription type and tenure have almost **no** effect on churn. Customers matching at least one of the
five main risk rules churn at **89.5%**; customers matching none churn at only **4.7%**.

### Top recommendations
1. **Escalate support cases at the 3rd–4th call**, before churn jumps at 5 calls.
2. **Act on late payments before day 20:** reminders, instalments, auto-pay incentives.
3. **Offer monthly customers a discount to move to a quarterly plan.**
4. **Reward customers for crossing the 500 spend mark.**
5. **Build a weekly early-warning list** from the risk rules, so retention budget targets the right customers.

See section 5 of the notebook for the full list, limitations and next steps.

---

## Mini Project 2 – Churn Prediction Model

**Question:** can a machine-learning model flag customers who are about to churn, so retention
promotions reach them in time?

### Files
- `mini_project_2_model.ipynb`: the notebook with all code, charts and observations
- `mini_project_2_model.py`: the same pipeline as a Python script (VS Code "percent" cell format)

### Pipeline
1. **EDA recap and data quality:** 1 empty row dropped; the "duplicate" IDs kept (two stacked sources, see Project 1)
2. **Preprocessing:** `customerid` and `source` excluded (not known for a new customer); 4 threshold flags
   engineered from the EDA; one-hot encoding and scaling inside a `ColumnTransformer`; no resampling (55/45 is balanced enough)
3. **Stratified 80/20 split**, with all preprocessing inside a scikit-learn `Pipeline` to avoid data leakage
4. **Models:** a rule-based baseline, Logistic Regression, and `HistGradientBoostingClassifier`, compared with 5-fold cross-validation
5. **Evaluation:** precision, recall, F1, ROC-AUC, confusion matrix, ROC/PR curves, threshold analysis, results per data source, permutation importance

### Results (test set, 101,042 customers)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline rule (any of 5 risk flags) | 0.919 | 0.896 | 0.966 | 0.930 | – |
| Logistic Regression | 0.888 | 0.902 | 0.895 | 0.899 | 0.941 |
| **Gradient Boosting** | **0.937** | 0.898 | **0.999** | **0.946** | **0.954** |

- **Gradient boosting** misses only **55** churners, against 1,898 for the simple rule. Recall was the main
  metric, because a lost customer costs more than an unnecessary promotion.
- **Logistic regression does not beat the simple rule**: churn follows "any risk condition → churn" logic,
  which trees capture and a single weighted sum does not.
- **Per-source check:** the model is near-perfect on source A (accuracy 0.999), but on source B it flags 97%
  of customers (accuracy ~0.50), even though it *ranks* them almost perfectly (ROC-AUC 0.999). The probabilities
  are on different scales for the two sources, so one threshold can't fit both. Very high overall scores on this
  dataset should therefore be read with care.
- **Top drivers** (permutation importance): support calls, payment delay, age, contract length, gender, which
  matches the EDA in Project 1.

See section 7 of the notebook for pros and cons, business use and improvements.

---

## How to run
**Colab:** open either Colab link above and choose *Runtime → Run all*. The dataset is downloaded from this repository automatically.

**Locally:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook
```
Then open `mini_project_1_eda.ipynb` or `mini_project_2_model.ipynb`.
