# ⚠️ Anomaly Detection: Quick Revision Notes

This summary covers the key formulas and steps for implementing the Gaussian distribution-based anomaly detection algorithm.

***

## 🎯 Problem & Dataset

| Component | Detail |
| :--- | :--- |
| **Goal** | Detect anomalous server behavior (failing servers). |
| **Features (2D)** | **Throughput** (mb/s) and **Latency** (ms). |
| **Training Set ($X_{\text{train}}$)** | Used to **fit the Gaussian distribution** ($\mu$ and $\sigma^2$). |
| **Cross-Validation Set ($X_{\text{val}}, y_{\text{val}}$)** | Used to **select the optimal threshold** ($\epsilon$). |
| **Core Idea** | Examples with **very low probability** ($p(x)$) under the fitted distribution are considered anomalies. |

***

## 📈 Gaussian Parameter Estimation (Exercise 1)

The goal is to estimate the mean ($\mu_j$) and variance ($\sigma^2_j$) for each feature ($j$) independently.

### **Formulas**

| Parameter | Formula (Univariate) | Vectorized Implementation |
| :--- | :--- | :--- |
| **Mean** ($\mu_j$) | $$\mu_j = \frac{1}{m} \sum_{i=1}^{m} x_j^{(i)}$$ | `mu = 1 / m * np.sum(X, axis = 0)` |
| **Variance** ($\sigma^2_j$) | $$\sigma_j^2 = \frac{1}{m} \sum_{i=1}^{m} (x_j^{(i)} - \mu_j)^2$$ | `var = 1 / m * np.sum((X - mu) ** 2, axis = 0)` |

### **Results (2D Dataset)**

* **Mean ($\mu$):** [14.112, 14.998]
* **Variance ($\sigma^2$):** [1.833, 1.710]
    

***

## 📊 Threshold Selection (Exercise 2)

The threshold ($\epsilon$) is selected by finding the value that maximizes the **F1 score** on the cross-validation set. An example $x$ is classified as an anomaly if $p(x) < \epsilon$.

### **F1 Score Formulas**

| Metric | Formula | Description |
| :--- | :--- | :--- |
| **Precision** ($P$) | $$P = \frac{\text{TP}}{\text{TP} + \text{FP}}$$ | Fraction of detected anomalies that are actually anomalous. |
| **Recall** ($R$) | $$R = \frac{\text{TP}}{\text{TP} + \text{FN}}$$ | Fraction of actual anomalies that were correctly detected. |
| **F1 Score** | $$\text{F1} = \frac{2 \cdot P \cdot R}{P + R}$$ | Harmonic mean of Precision and Recall. |

### **Results (2D Dataset)**

* **Best Epsilon ($\epsilon$):** $8.99 \times 10^{-5}$
* **Best F1 Score:** $0.875$

***

## 🚀 High-Dimensional Application

The same Gaussian anomaly detection algorithm was applied to a complex dataset with 11 features.

| Component | Value |
| :--- | :--- |
| **Number of Features ($n$)** | 11 |
| **Best Epsilon ($\epsilon$)** | $1.38 \times 10^{-18}$
| **Best F1 Score** | $0.615$
| **Anomalies Found** | 117 |
