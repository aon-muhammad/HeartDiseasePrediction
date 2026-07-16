
# ❤️ Heart Disease Prediction using Machine Learning

## 📌 Overview

This project predicts whether a patient is at risk of cardiovascular disease using supervised machine learning algorithms. The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, model comparison, and evaluation.

The objective is to compare multiple classification algorithms and identify the model that performs best on the dataset.

https://aon-muhammad-heartdiseaseprediction-app-xvob8g.streamlit.app/ Streamlit link

## 📂 Dataset

* **Dataset:** Cardiovascular Disease Dataset
* **Samples:** ~70,000
* **Target Variable:** `cardio`

  * `0` = No cardiovascular disease
  * `1` = Cardiovascular disease

### Features

* Age
* BMI (Feature Engineered)
* Systolic Blood Pressure (`ap_hi`)
* Diastolic Blood Pressure (`ap_lo`)
* Cholesterol Level
* Glucose Level
* Smoking Status
* Alcohol Consumption
* Physical Activity

---

## 📊 Exploratory Data Analysis

The following analyses were performed:

* Missing value inspection
* Outlier detection
* Correlation analysis
* Distribution plots
* Feature relationship analysis
* Class distribution analysis

---

## 🧹 Data Preprocessing

The preprocessing pipeline included:

* Removal of invalid blood pressure values
* Removal of unrealistic observations
* Feature engineering by creating **BMI**
* Dropping redundant features (`height`, `weight`, and `gender`)
* One-Hot Encoding of categorical variables
* Standardization of continuous numerical features using `StandardScaler`
* Train-test split

---

## 🤖 Machine Learning Models

The following classification algorithms were evaluated:

* Logistic Regression
* Gaussian Naive Bayes
* Decision Tree Classifier
* K-Nearest Neighbors (KNN)

---

## 📈 Model Performance

| Model                |   Accuracy |   F1 Score |
| -------------------- | ---------: | ---------: |
| Logistic Regression  | **73.38%** | **0.7121** |
| Gaussian Naive Bayes |     69.84% |     0.6551 |
| Decision Tree        |     64.29% |     0.6339 |
| K-Nearest Neighbors  |     69.64% |     0.6894 |

### Best Model

**Logistic Regression** achieved the best overall performance and was selected as the final model.

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib
* Streamlit

---

## 📁 Project Structure

```
Heart-Disease-Prediction/
│
├── data/
├── notebook.ipynb
├── app.py
├── heart_disease_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

## 🚀 Future Improvements

* Hyperparameter tuning
* Ensemble learning (Random Forest, Gradient Boosting)
* Model deployment using Streamlit Cloud
* Improved feature engineering
* Cross-validation for robust model evaluation

---

## 👨‍💻 Author

**Aon Muhammad**

This project was developed as part of my Machine Learning learning journey to strengthen my understanding of data preprocessing, feature engineering, model evaluation, and deployment.
