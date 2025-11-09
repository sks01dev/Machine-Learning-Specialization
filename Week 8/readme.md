# ❤️ Heart Failure Prediction: Ensemble Classification (C2, Week 4 Lab)

This document summarizes the data preparation, model training, and hyperparameter investigation performed in the **C2\_W4\_Lab\_02\_Tree\_Ensemble.ipynb** notebook. The goal is to predict the likelihood of **Heart Disease** ($\mathbf{1}$ or $\mathbf{0}$) using a public health dataset.

---

## I. Data Preparation

### **1. Objective and Dataset**
* **Objective:** Binary **Classification**—predicting the `HeartDisease` status of a patient.
* **Dataset:** Heart Failure Prediction Dataset (obtained from Kaggle).
* **Evaluation Metric:** **Accuracy Score**.

### **2. One-Hot Encoding**
Since Decision Trees and most machine learning models require numerical inputs, Pandas was used to **One-Hot Encode** the categorical features:
* `Sex`
* `ChestPainType`
* `RestingECG`
* `ExerciseAngina`
* `ST_Slope`

### **3. Data Splitting**
The processed data was split using scikit-learn's `train_test_split`:
* **Training Set** ($\approx 80\%$ of data)
* **Validation Set** ($\approx 20\%$ of data)

---

## II. Model Implementation and Hyperparameter Tuning

The core of the lab was comparing a single Decision Tree against two powerful ensemble classifiers.

### **1. Decision Tree Classifier**

Hyperparameter tuning was manually performed by observing the behavior of the training and validation accuracy when adjusting model complexity.

| Hyperparameter | Effect of Tuning | Key Takeaway |
| :--- | :--- | :--- |
| **`min_samples_split`** | Increasing this value reduces overfitting by demanding more samples before a node can be split. | Increasing this value brought training accuracy closer to validation accuracy. |
| **`max_depth`** | Decreasing this value controls the size and complexity of the tree. | A high value ($\ge 5$) led to overfitting; a low value ($\le 3$) led to underfitting. The optimal was around 4. |

### **2. Random Forest Classifier (Bagging)**

The Random Forest, an ensemble of Decision Trees, was tested to reduce variance.

* **Key Hyperparameter:** **`n_estimators`** (the number of trees).
* **Observation:** Increasing the number of estimators reduced the gap between training and validation accuracy, minimizing overfitting. This is a characteristic benefit of **Bagging** methods.

### **3. XGBoost Classifier (Boosting)**

XGBoost was implemented as a powerful **Boosting** ensemble method to further minimize prediction error (bias).

* **Key Mechanism: Early Stopping**
    * The model uses the `eval_set` (validation data) and `early_stopping_rounds` parameters during fitting.
    * A large number of `n_estimators` is initially set (e.g., 500).
    * Training automatically stops if the evaluation metric (`logloss`) on the validation set does not improve for a set number of rounds (e.g., 10).
    * This technique effectively limits the number of trees created, helping to **reduce overfitting** while maximizing performance on unseen data.

---

## III. Final Performance Comparison

The final fitted ensemble models achieved similar strong performance on the validation set:

| Model | Final Train Accuracy | Final Validation Accuracy |
| :--- | :--- | :--- |
| **Random Forest** | $0.9332$ | $\mathbf{0.8859}$ |
| **XGBoost** | $0.9251$ | $0.8641$ |

Both ensemble methods outperformed the single Decision Tree model, successfully minimizing the bias-variance trade-off for this classification task.

