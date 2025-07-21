# 💎 Diamond Prices Prediction using SVM Regressor & GridSearchCV

### Support Vector Regression + Hyperparameter Tuning with GridSearchCV
This notebook presents a complete pipeline to **predict diamond prices** using **Support Vector Machine (SVM) Regression**, with integrated **GridSearchCV** for hyperparameter tuning. It includes data exploration (EDA), preprocessing, model training, and evaluation.

📌 **Kaggle Dataset**: [Dataset](https://www.kaggle.com/datasets/shivam2503/diamonds)

---

## 📂 Dataset Overview

The dataset contains information on **53,920 diamonds** with their physical properties and market price. It’s a well-known dataset for regression modeling and machine learning experiments.

### 📋 Features Table

| Feature Name | Description |
|--------------|-------------|
| `carat`      | Weight of the diamond (in carats) |
| `cut`        | Quality of the cut (e.g., Fair, Good, Ideal) |
| `color`      | Diamond color grade (J to D, worst to best) |
| `clarity`    | Diamond clarity level (I1, SI2, ..., IF) |
| `depth`      | Depth percentage (z / mean(x, y)) × 100 |
| `table`      | Width of top of diamond relative to widest point |
| `price`      | Price of the diamond in USD |
| `x`          | Length in mm |
| `y`          | Width in mm |
| `z`          | Depth in mm |

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was conducted to understand the structure and patterns in the dataset:

- **Boxplots** for detecting outliers in numeric columns
- **Correlation Analysis** between `price` and other numerical features
- **Categorical counts and distributions** for `cut`, `color`, `clarity`
- **Missing value inspection** and general shape analysis

---

## 🧹 Preprocessing

- **Label Encoding** for categorical variables (`cut`, `color`, `clarity`)
- **Feature Scaling** using `StandardScaler`
- **Outlier detection** via visualizations
- Split: `train_test_split()` with 80% train and 20% test sets

---

## 🔍 Hyperparameter Tuning

 - Used `GridSearchCV` to find the best combination of parameters
   - Parameter grid:

     ```python
     param_grid = {
         'C': [0.1, 1, 10, 100, 1000],
         'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
         'kernel': ['rbf', 'linear']
     }
     ```

---


## ⚙️ Model Training - SVM with GridSearchCV

The notebook implements **Support Vector Regressor (SVR)** and optimizes its performance using **GridSearchCV** to find the best combination of hyperparameters:

### 🎯 Parameters Tuned:
- `C` (Regularization)
- `gamma` (Kernel coefficient)
- `kernel` (e.g., linear, rbf, poly)

```python
from sklearn.model_selection import GridSearchCV
params = {'C': [...], 'gamma': [...], 'kernel': [...]}
grid = GridSearchCV(SVR(), param_grid=params, cv=5)
grid.fit(X_train, y_train)
