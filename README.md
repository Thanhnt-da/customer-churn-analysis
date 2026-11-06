# Customer Churn Analysis

An e-commerce company wants to predict which users will churn so it can offer them targeted
promotions. This repository contains the two mini projects built on the same customer dataset.

| Project | Scope | Notebook | Colab |
|---|---|---|---|
| **Mini Project 1 – EDA** | Data understanding, cleaning, descriptive statistics, relationship analysis, business recommendations | [`mini_project_1_eda.ipynb`](mini_project_1_eda.ipynb) | [Open in Colab](COLAB_LINK_PROJECT_1) |
| **Mini Project 2 – Churn Model** | Supervised learning model to predict churn | *coming soon* | *coming soon* |

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

### How to run
**Colab:** open the Colab link above and choose *Runtime → Run all*. The dataset is downloaded from this repository automatically.

**Locally:**
```bash
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook mini_project_1_eda.ipynb
```
