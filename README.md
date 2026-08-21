# Bank Marketing Campaign — Deposit Subscription Prediction

## Project Overview

The **Bank Marketing Campaign** project is a Machine Learning application designed to predict whether a customer is likely to subscribe to a **term deposit**.

The main goal is to help banks improve their marketing campaigns by identifying customers with a higher probability of subscribing, allowing marketing teams to prioritize their efforts instead of contacting every customer.

The project combines exploratory data analysis, data preprocessing, feature engineering, multiple Machine Learning models, ensemble learning, and decision-threshold optimization.

---

## Problem Statement

Bank marketing campaigns often require a large number of customer calls, while many contacted customers do not subscribe to the offered term deposit.

This creates unnecessary marketing costs and effort.

The objective of this project is to build a predictive system that estimates whether a customer is likely to subscribe **before the marketing call**, helping the bank focus its resources on customers with a higher likelihood of saying yes.

---

## Project Objectives

* Predict customer term-deposit subscription.
* Identify customers with a high probability of subscription.
* Compare multiple Machine Learning algorithms.
* Optimize models according to the business objective.
* Prioritize **Recall** to reduce missed potential customers.
* Build an ensemble model using Random Forest and LightGBM.
* Optimize the classification decision threshold.
* Deploy the final solution as a Streamlit web application.

---

## Team

This project was developed collaboratively by:

* Fatma
* Hager
* Rehab
* Marwan
* Abubakr

---

## Dataset

The project uses a dataset named **`bank.csv`** from a Portuguese bank.

### Dataset Characteristics

* **Records:** 11,162 customers
* **Features:** 17
* **Target:** `deposit`
* **Target values:** `yes` / `no`

The features can be grouped into three main categories:

### Customer Information

* Age
* Job
* Marital status
* Education
* Balance
* Loans

### Current Campaign Information

* Contact method
* Day
* Month
* Call duration
* Number of campaign contacts

### Previous Campaign Information

* Days since the previous contact
* Number of previous contacts
* Previous campaign outcome

---

## Exploratory Data Analysis — Key Findings

Several important patterns were identified during the exploratory analysis.

### Unknown Categories

Some categorical features such as `job`, `education`, and `contact` contain `"unknown"` values instead of conventional missing values.

These values were treated as valid categorical classes rather than being removed.

### `pdays` Interpretation

The `pdays` feature contains `-1` for customers who had not been contacted previously.

Because `-1` does not represent an actual number of days, a separate flag was created to represent whether the customer had been contacted before.

### Target Distribution

The target variable is approximately balanced between subscribers and non-subscribers.

Therefore, **Accuracy alone is not sufficient** to evaluate the models, and additional metrics such as Recall, Precision, F1-score, and ROC-AUC were considered.

### Job Type

Customer occupation showed noticeable differences in subscription rates.

Students and retired customers showed higher subscription rates, while workers and entrepreneurs showed lower subscription rates.

### Previous Campaign Outcome

Customers who subscribed during previous campaigns showed a higher likelihood of subscribing again.

---

## Data Preprocessing & Feature Engineering

The preprocessing pipeline included several steps designed to improve data quality while avoiding data leakage.

### 1. `pdays` Feature Engineering

A flag was created to identify customers who had previously been contacted.

The original `-1` value was replaced with `0` after creating the corresponding indicator.

### 2. Outlier Handling

Extreme values in the number of campaign contacts were reduced using a **99th-percentile cap** rather than deleting the affected records.

### 3. Encoding

Binary `yes`/`no` features were converted into `1`/`0`.

Categorical features were transformed using **One-Hot Encoding**.

### 4. Train/Test Split

The dataset was split into:

* **80% Training**
* **20% Testing**

The split was performed before encoding and scaling to help prevent data leakage.

### 5. Feature Scaling

`StandardScaler` was fitted only on the training data and then applied to the test data.

---

## Machine Learning Models

Eight model configurations were evaluated:

1. Logistic Regression — Default
2. Logistic Regression — Balanced
3. KNN — Tuned
4. Decision Tree — Final
5. Random Forest — Recall Optimized
6. XGBoost — Recall Optimized
7. LightGBM — Recall Optimized
8. Random Forest + LightGBM Ensemble

### Model Tuning

Different hyperparameters were explored depending on the algorithm.

Examples include:

* KNN: number of neighbors and distance metrics
* Decision Tree: tree depth and split parameters
* Random Forest: hyperparameter optimization
* XGBoost: class weighting
* LightGBM: gradient boosting configuration

---

## Model Performance Comparison

| Model                            |  Accuracy | Precision |    Recall |  F1-Score |   ROC-AUC |
| -------------------------------- | --------: | --------: | --------: | --------: | --------: |
| Logistic Regression (Default)    |     82.7% |     82.7% |     80.2% |     81.4% |     0.907 |
| Logistic Regression (Balanced)   |     82.9% |     82.1% |     81.7% |     81.9% |     0.907 |
| KNN (Tuned)                      |     79.3% |     78.9% |     76.8% |     77.8% |     0.874 |
| Decision Tree (Final)            |     81.5% |     78.6% |     83.7% |     81.0% |     0.892 |
| Random Forest (Recall-Optimized) |     85.8% |     81.5% |     90.6% |     85.8% |     0.923 |
| XGBoost (Recall-Optimized)       | **87.0%** | **83.8%** |     90.0% | **86.8%** |     0.928 |
| LightGBM (Recall-Optimized)      |     86.4% |     82.7% |     90.2% |     86.3% | **0.932** |
| **RF + LightGBM Ensemble**       |     86.8% |     80.3% | **95.8%** | **87.3%** |     0.929 |

---

## Final Model — Random Forest + LightGBM Ensemble

The final solution combines **Random Forest** and **LightGBM** using **Soft Voting**.

The probability generated by each model is averaged:

```python
ensemble_proba = (rf_proba + lgbm_proba) / 2
```

This produces a combined subscription probability for each customer.

### Final Ensemble Performance

* **Accuracy:** 86.8%
* **Precision:** 80.3%
* **Recall:** 95.8%
* **F1-Score:** 87.3%
* **ROC-AUC:** 0.929

The ensemble was selected as the final model because it achieved the **highest Recall** among the evaluated models.

---

## Why Recall Matters

Recall was selected as the primary business metric.

In this scenario, failing to identify a customer who is actually likely to subscribe can represent a missed marketing opportunity.

On the other hand, contacting a customer who ultimately does not subscribe mainly results in an additional marketing interaction.

Therefore, the project prioritizes identifying as many potential subscribers as possible while maintaining acceptable overall performance.

The final ensemble achieved a **95.8% Recall**, meaning it correctly identified approximately **96 out of every 100 actual subscribers** in the evaluated test set.

---

## Decision Threshold Optimization

Instead of relying on the default classification threshold of `0.50`, multiple thresholds were evaluated from `0.10` to `0.90`.

The final selected threshold was:

### **0.35 — 35%**

A customer is classified as a potential subscriber when:

```text
Ensemble Probability >= 0.35
```

Otherwise:

```text
Ensemble Probability < 0.35
```

The threshold was selected based on the project's focus on maximizing Recall while maintaining an Accuracy of at least 70%.

---

## Machine Learning Pipeline

```text
Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Train / Test Split
   ↓
Encoding
   ↓
Feature Scaling
   ↓
Model Training & Tuning
   ↓
Model Evaluation
   ↓
Random Forest + LightGBM
   ↓
Soft Voting
   ↓
Threshold Optimization
   ↓
Final Prediction
```

---

## Streamlit Application

The trained models were integrated into a **Streamlit** web application.

The application allows users to enter customer information and receive:

* Subscription prediction
* Subscription probability
* Recommended marketing action

The application uses the trained Random Forest and LightGBM models to generate the final prediction.

---

## Technologies Used

* **Python**
* **Pandas**
* **Scikit-learn**
* **LightGBM**
* **Joblib**
* **Streamlit**
* **Git**
* **GitHub**

---

## Project Structure

```text
Bank-Marketing-Campaign/
│
├── app.py
├── requirements.txt
│
└── models/
    ├── rf_model.pkl
    ├── lgbm_model.pkl
    └── scaler.pkl
```

---

## How to Run Locally

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will then be available locally through the Streamlit server.

---

## Live Demo

The deployed application is available here:

**[Bank Marketing Campaign — Live Demo](https://bank-marketing-campaign-jjihnokumwpjf3wwi4vlre.streamlit.app/)**

---

## Presentation

The project presentation will be included in the repository to provide an overview of the problem, methodology, Machine Learning models, evaluation results, and final solution.

---

## Screenshots

Screenshots of the deployed Streamlit application will be added here.

---

## Conclusion

The **Bank Marketing Campaign** project demonstrates an end-to-end Machine Learning workflow for customer subscription prediction.

Multiple classification algorithms were trained, tuned, and evaluated using several performance metrics. The final **Random Forest + LightGBM Soft Voting Ensemble** achieved the highest Recall of **95.8%**, making it the selected model for the project's marketing objective.

The final solution was deployed as an interactive Streamlit application, transforming the Machine Learning workflow into a practical customer prediction tool.
