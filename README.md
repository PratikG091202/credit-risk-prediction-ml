# Credit Risk Analysis

This repository contains a machine learning workflow for **consumer credit risk evaluation** and **loan default prediction** (Fully Paid vs. Charged Off).  
The project leverages a Kaggle dataset (`zaurbegiev/my-dataset`) with **100,514 lending records** and applies systematic data preprocessing, anomaly detection, and model training.

---

## 📂 Dataset
- Source: Kaggle (`zaurbegiev/my-dataset`)
- Records: 100,514 entries, 19 columns
- Target: **Loan Status** (Fully Paid vs. Charged Off)

Key features include:
- Loan Amount, Term, Credit Score, Annual Income
- Employment length (Years in current job)
- Home Ownership, Purpose
- Monthly Debt, Credit History, Credit Problems
- Bankruptcies, Tax Liens

---

## 🛠️ Data Preprocessing
1. **Deduplication** → Removed **10,728 duplicate records**.  
2. **Target Integrity** → Dropped rows with missing Loan Status.  
3. **Feature Pruning** → Removed identifiers (`Loan ID`, `Customer ID`) and `Months since last delinquent`.  
4. **Ordinal Transformation** → Converted `Years in current job` into integers (0–10).  
5. **Outlier Treatment** →  
   - Removed placeholder loan amounts (`99,999,999`).  
   - Corrected inflated credit scores (>850) by dividing by 10.  
6. **Imputation Strategy** →  
   - Numerical features → median values.  
   - Categorical features → mode values.  

---

## 📊 Exploratory Data Analysis (EDA)
- **Loan Status Distribution** → Imbalanced dataset (more Fully Paid loans).  
- **Loan Term** → Short vs. long-term loans.  
- **Home Ownership** → Mortgage, Own, Rent categories.  
- **Credit Score** → Detected anomalies (>850).  
- **Current Loan Amount** → Placeholder values identified and removed.  
- **Boxplots & Histograms** → Used to detect anomalies in income, debt, and loan amounts.  

---

## 🔬 Modeling

### Support Vector Machine (SVM)
- Standardized features with **StandardScaler**.  
- Tuned via **GridSearchCV** (3-fold stratified CV).  
- Hyperparameters: `C ∈ {0.001, 0.01, 0.1, 1, 10}`, Kernel = RBF, Class Weight = balanced.  
- Optimal Layout: `{'C': 0.001, 'class_weight': 'balanced', 'kernel': 'rbf'}`.  
- Integrated **Platt Scaling** for probability outputs.  

### Extreme Gradient Boosting (XGBoost)
- Trained on raw unscaled features for interpretability.  
- Addressed imbalance with `scale_pos_weight = 2.458`.  
- Manual brute-force grid search across **275 parameter combinations**.  
- Optimal Layout: `{'n_estimators': 250, 'max_depth': 4, 'learning_rate': 0.1}`.  

---

## 📈 Performance Summary

| Model Framework              | Accuracy | Precision | Recall (Default) | ROC-AUC |
|-------------------------------|----------|-----------|------------------|---------|
| Baseline RBF SVM              | 60.39%   | 0.375     | 55.52%           | 0.590   |
| Tuned RBF SVM (GridSearchCV)  | 65.44%   | 0.402     | 40.08%           | 0.609   |
| Baseline XGBoost              | 60.00%   | 0.370     | 54.00%           | 0.580   |
| Optimized XGBoost (Loops)     | 61.00%   | 0.390     | 59.00%           | 0.649   |

---

## 📌 Conclusions
- **Optimized XGBoost (AUC = 0.65)** outperforms Tuned SVM (AUC = 0.61).  
- XGBoost achieves **higher recall (59%)**, making it the optimal tool for minimizing default risk.  
- SVM provides stable metrics but struggles with local sensitivity due to reliance on global distance matrices.  
- Business impact: XGBoost better captures **default risk exposure**, reducing institutional capital impairment.  

---

## 🚀 Getting Started

### Prerequisites
Install dependencies:
pip install numpy pandas matplotlib seaborn scikit-learn xgboost kagglehub>=1.0.0
