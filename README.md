# 🏡 Ames Housing Price Prediction & Kaggle Challenge

## 📌 Overview
This project develops a **regression model** to predict house sale prices using the **Ames Housing Dataset**. The model is evaluated using **cross-validation and Kaggle submissions**, refining predictions iteratively.

## 🎯 Problem Statement
In real estate, accurately pricing homes is crucial for buyers, sellers, and realtors. Mispricing can lead to **financial losses**, either by **overpricing (longer market times)** or **underpricing (lost revenue)**. 

This project aims to:
- **Predict house sale prices** using key property features.
- **Identify factors and Model Evaluation** that significantly impact home value.
- **Provide data-driven insights** to help stakeholders make informed real estate decisions.

---

## 📂 Dataset
The **Ames Housing Dataset** consists of detailed property information with **over 70 features**.

- **Train (`train.csv`)** – Includes `SalePrice` (target variable).
- **Test (`test.csv`)** – No `SalePrice`, used for Kaggle submission.
- **Submission (`submission.csv`)** – Final predictions formatted for Kaggle.

### 🔹 **Key Features Used**
| Feature | Description |
|---------|------------|
| `Overall Qual` | Overall material & finish quality |
| `Gr Liv Area` | Above-grade living area (sq. ft.) |
| `Garage Cars` | Number of garage spaces |
| `Garage Area` | Garage Area (sq. ft.) |
| `Year Built` | Year the house was built |
---

## 🛠️ Data Preprocessing
1. **Handled missing values** (imputed `0` for numerical columns, encoded categorical variables).
2. **Feature selection & transformation** (e.g., `log(SalePrice)` to handle skewed data).
4. **Ensured consistency** across training and test datasets.

---

## 📈 Model Training & Evaluation
We trained multiple regression models and evaluated performance using **R² Score, RMSE, and MAE**.
R² Score: 0.7935 (79.35%)
Mean Absolute Error (MAE): 25918.57
Mean Squared Error (MSE): 1227109428.11
Root Mean Squared Error (RMSE): 35030.12

Despite R2 score explaining quite good portion of variance model has still high percentage of errors which indicates model being Underfitting. 

### 🔹 **Models Tested**
✅ **Baseline Model:** Simple Linear Regression using only `Overall Qual`  
✅ **Feature-Enhanced Regression:** Multi-feature Linear Regression  
✅ **Regularized Models:** Ridge & Lasso Regression for better generalization  

| Model |        R²  |  RMSE |    |
|--------|------------|------------|---
 Linear Regression | 0.79 | 35030.12  

---

## 📤 Kaggle Submission
Predictions were generated and submitted to **Kaggle's DSB-210 Regression Challenge**.

```python
# Train the final model
lr.fit(X_train, y_train)

# Make predictions
y_test_pred = lr.predict(X_test)

# Save submission file
submission = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': y_test_pred})
submission.to_csv('submission.csv', index=False)
