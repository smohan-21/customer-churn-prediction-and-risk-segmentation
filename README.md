# customer-churn-prediction-and-risk-segmentation

# 📉 Customer Churn Prediction & Risk Segmentation

> **Week 2 Project | AI & Data Analyst Internship**

An end-to-end machine learning pipeline that predicts customer churn for a telecom business, segments customers into risk tiers, and delivers business-ready insights through interactive visualizations.

---

## 📌 Problem Statement

Subscription-based businesses lose significant revenue every year due to customer churn — customers who cancel or stop using a service. **Predicting which customers are likely to churn before they actually leave** allows the business to act proactively: offer discounts, improve support, or tailor communication.

This project simulates a real analyst task at a product or growth team, with a focus on **interpretability** — so that a business stakeholder can understand the findings, not just the code.

---

## 📁 Repository Structure

```
customer-churn-prediction-and-risk-segmentation/
│
├── Analysis.ipynb            # Main Colab Notebook (all 6 tasks)
├── Model_Comparison.png      # Side-by-side model performance comparison chart
├── requirements.txt          # All required Python libraries
├── summary.pdf               # Final insights & business recommendations report
├── Dataset/                  # Raw telecom customer churn CSV dataset
└── Charts/                   # All generated visualization charts
```

---

## ✅ Tasks Completed

### Task 1 — Data Loading & Exploratory Analysis
- Loaded the telecom dataset using **Pandas** and displayed shape, column types, and first 10 rows
- Identified the target column (`Churn` — Yes/No) and checked for **class imbalance**
- Found and handled missing/null values (drop vs. impute decision made)
- Computed summary statistics for numerical columns (mean, median, std)
- Plotted a **correlation heatmap** of numerical features using Seaborn

### Task 2 — Data Preprocessing & Feature Engineering
- Converted `TotalCharges` to numeric with careful coercion error handling
- Encoded all categorical columns using **Label Encoding / One-Hot Encoding** (with justification)
- Created 2 new engineered features:
  - `ChargesPerMonth` = TotalCharges / tenure
  - `SeniorWithNoSupport` = (SeniorCitizen == 1) & (TechSupport == "No")
- Scaled numerical features using **StandardScaler**
- Split data into **80% train / 20% test** using stratified split to preserve class ratio

### Task 3 — Model Training & Comparison
- Trained 3 classification models:
  - **Model 1:** Logistic Regression
  - **Model 2:** Random Forest Classifier
  - **Model 3:** XGBoost / Gradient Boosting Classifier
- Evaluated all models with: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Plotted **Confusion Matrix** for each model side-by-side
- Plotted **ROC Curve** for all 3 models on one chart for comparison
- Performed hyperparameter tuning on the best model using **GridSearchCV / RandomizedSearchCV** *(Advanced)*

### Task 4 — Customer Risk Segmentation *(Advanced)*
- Used the best model's **predicted churn probability** (not just 0/1 labels)
- Segmented all customers into 3 risk tiers:
  - 🔴 **High Risk** → probability ≥ 0.70
  - 🟡 **Medium Risk** → probability between 0.40 and 0.69
  - 🟢 **Low Risk** → probability < 0.40
- Counted customers per tier and computed per-tier averages:
  - Average `MonthlyCharges`, average `tenure`, and `Contract` type distribution
- Created a **grouped bar chart** comparing the 3 tiers across key features

### Task 5 — Visualizations (4 Charts)
- **Chart 1:** Feature importance bar chart (top 10 features) from Random Forest / XGBoost
- **Chart 2:** Churn rate by Contract Type (Month-to-Month vs. One Year vs. Two Year)
- **Chart 3:** Distribution of tenure — churned vs. not-churned customers (KDE / overlapping histogram)
- **Chart 4:** Risk tier donut/pie chart showing customer count per tier
- **Bonus Chart:** Interactive scatter plot using **Plotly** — MonthlyCharges vs. Tenure, colored by churn label

All charts are saved in the `Charts/` folder. Model comparison chart: [`Model_Comparison.png`](./Model_Comparison.png)

### Task 6 — Insights & Business Recommendations
A structured analysis written inside the notebook covering:
- Which model performed best and why it was selected
- Top 3 factors driving customer churn
- What High Risk customers have in common (contract type, charges, tenure)
- 2 specific, actionable business recommendations
- Model limitations and future improvement suggestions

Full written summary available in [`summary.pdf`](./summary.pdf).

---

## 🛠️ Tools & Libraries Used

| Tool | Purpose |
|---|---|
| Python 3.x | Main programming language |
| Google Colab | Development environment |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Models, metrics, preprocessing |
| XGBoost | Gradient boosting model |
| Matplotlib / Seaborn | Static charts and heatmaps |
| Plotly | Interactive visualizations |

Install all dependencies at once:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run This Project

**1. Open in Google Colab**

Click below to open the notebook directly — no setup needed:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smohan-21/customer-churn-prediction-and-risk-segmentation/blob/main/Analysis.ipynb)

**Or manually:**
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Open Notebook → GitHub**
3. Paste: `https://github.com/smohan-21/customer-churn-prediction-and-risk-segmentation`
4. Select `Analysis.ipynb`

**2. Install required libraries** (run this cell first in Colab)
```python
!pip install -r requirements.txt
```

**3. Upload or mount the dataset**
```python
from google.colab import drive
drive.mount('/content/drive')
```

Or upload the CSV manually from the `Dataset/` folder when prompted.

Run all cells from top to bottom to reproduce the full analysis.

---

## 📊 Key Outputs

- Cleaned & feature-engineered dataset with churn probability scores
- 3 trained and evaluated ML models with full metrics comparison
- Customer risk segmentation table (High / Medium / Low)
- 4+ visualizations saved in the `Charts/` folder
- Model comparison chart: [`Model_Comparison.png`](./Model_Comparison.png)
- Business insights report: [`summary.pdf`](./summary.pdf)

---

## 👤 Author

**Mohan S**
AI & Data Analyst Intern — Week 2 Project
GitHub: [@smohan-21](https://github.com/smohan-21)

---

## 📄 License

This project is built for internship and educational purposes.
