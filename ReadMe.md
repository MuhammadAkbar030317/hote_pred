# 🏨 Hotel Booking Cancellation Prediction (Classification)

An **end-to-end machine learning classification project** that predicts whether a hotel booking will be **cancelled or not**, based on customer, booking, and hotel-related features.
This project is designed with **clean architecture**, **data leakage prevention**, and **production-readiness** in mind.

---

## 📌 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset Description](#dataset-description)
4. [Target Variable](#target-variable)
5. [Project Workflow](#project-workflow)
6. [Feature Engineering](#feature-engineering)
7. [EDA (Exploratory Data Analysis)](#eda-exploratory-data-analysis)
8. [Baseline Model](#baseline-model)
9. [Model Improvement](#model-improvement)
10. [Train / Test Strategy](#train--test-strategy)
11. [Avoiding Data Leakage](#avoiding-data-leakage)
12. [Model Training & Evaluation](#model-training--evaluation)
13. [Inference & Prediction](#inference--prediction)
14. [Project Structure](#project-structure)
15. [Installation & Usage](#installation--usage)
16. [Technologies Used](#technologies-used)
17. [Future Improvements](#future-improvements)

---

## 📖 Project Overview

Hotel booking cancellations cause **significant revenue loss** in the hospitality industry.
This project aims to **predict booking cancellation in advance**, allowing hotels to:

* Optimize room allocation
* Reduce overbooking risk
* Improve revenue management strategies

The problem is modeled as a **binary classification task**.

---

## ❓ Problem Statement

> Given booking-related information, predict whether a hotel booking will be **cancelled (1)** or **not cancelled (0)**.

---

## 📊 Dataset Description

The dataset contains historical hotel booking records, including:

* Customer information
* Booking details
* Stay duration
* Pricing and deposit type
* Previous cancellation history

Each row represents **one booking**.

---

## 🎯 Target Variable

| Variable Name | Description                                        |
| ------------- | -------------------------------------------------- |
| `is_canceled` | 1 → Booking cancelled 0 → Booking not cancelled    |

This is a **binary classification problem**.

---

## 🔄 Project Workflow

1. Data Loading
2. Train / Test Split
3. Data Cleaning
4. Exploratory Data Analysis (EDA)
5. Feature Engineering
6. Baseline Model Training
7. Model Optimization
8. Model Evaluation
9. Inference Pipeline

---

## 🛠 Feature Engineering

Key feature engineering steps include:

* Handling missing values
* Encoding categorical variables
* Scaling numerical features
* Dropping leakage-prone columns
---

## 📈 EDA (Exploratory Data Analysis)

EDA helps to:

* Understand cancellation patterns
* Identify important predictors
* Detect outliers and skewed distributions

Visualizations include:

* Cancellation rate by hotel type
* Lead time vs cancellation
* Deposit type impact
* Previous cancellations analysis

---

## ⚙️ Baseline Model

A simple baseline model is used for initial benchmarking:

* Logistic Regression **or**
* Decision Tree Classifier

This helps evaluate whether advanced models bring real improvements.

---

## 🚀 Model Improvement

Advanced models used:

* Random Forest
* XGBoost / LightGBM
* Hyperparameter tuning using Optuna

---

## 🧪 Train / Test Strategy

* Stratified train-test split
* Typical split ratio: **80 / 20**

---

## 🔒 Avoiding Data Leakage

To prevent data leakage:

* Target-related features removed before training
* Feature engineering applied **after** train-test split
* Pipelines used for preprocessing + modeling

---

## 📊 Model Training & Evaluation

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-score

Confusion matrix used for detailed error analysis.

---

## 🔍 Inference & Prediction

The trained model can predict cancellation probability for **new bookings**:

* Input: Booking details
* Output: Cancellation probability + class label

---

## 📁 Project Structure

```
├── .github/
│   └── workflows/          # CI/CD pipelines (GitHub Actions)
│
├── app/                    # Application entry point (API / service)
│
├── data/                   # Datasets (raw)
│
├── models/
│   └── baseline/bestmodel  # Saved baseline models & checkpoints
│
├── notebooks/              # Jupyter notebooks (EDA, baseline_model)
│
├── scripts/                #  calling classes
│
├── src/                    # Classes
│
├── tests/                  # Testing
│
├── .dockerignore           # Exclude files from Docker image
├── .gitignore              # Exclude files from Git tracking
├── Dockerfile              # Docker build configuration
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ▶️ Installation & Usage

```bash
# create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run training
python main.py
```

---

## 🧰 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Jupyter Notebook
* Joblib

---

## 🔮 Future Improvements

* Deploy model using FastAPI
* Add real-time inference API
* Model monitoring & drift detection
* Explainability using SHAP

---

## ✍️ Author

**Abdurayimov Muhammad Akbar**
Supervised Machine Learning Project

---

✅ **Task Type:** Classification
✅ **Use Case:** Hotel Booking Cancellation Prediction


