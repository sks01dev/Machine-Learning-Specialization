# Advice for Applying Machine Learning 

This document summarizes the key concepts and techniques used in the **C2\_W3\_Assignment.ipynb** notebook for **evaluating and improving machine learning models** in the context of bias and variance trade-offs.

-----

## 1\. Data Splitting for Evaluation

A model must be evaluated on data it hasn't seen to gauge its performance on *new* examples.

### **Three-Set Split**

To tune a model while accurately estimating its generalization error, data is split into three sets:

| Dataset | Typical Split | Purpose |
| :--- | :--- | :--- |
| **Training Set** | 60% | Used to **fit** the model parameters ($\mathbf{w}, b$) (training/fitting). |
| **Cross-Validation (CV) Set** | 20% | Used to **tune** hyperparameters (e.g., polynomial degree, regularization parameter $\lambda$). |
| **Test Set** | 20% | Used for a **final, unbiased evaluation** of the tuned model's performance on new data. |

-----

## 2\. Error Calculation

### **Mean Squared Error (MSE) for Regression**

Used to evaluate linear/polynomial regression models on the test or CV set:

$$J_{\text{test}}(\mathbf{w},b) = \frac{1}{2m_{\text{test}}}\sum_{i=0}^{m_{\text{test}}-1} ( f_{\mathbf{w},b}(\mathbf{x}^{(i)}_{\text{test}}) - y^{(i)}_{\text{test}} )^2$$

### **Classification Error for Categorical Models**

Used to evaluate neural network classification models: the fraction of incorrect predictions.

$$J\_{cv} = \\frac{1}{m}\\sum\_{i=0}^{m-1}
\\begin{cases}
1, & \\text{if $\hat{y}^{(i)} \neq y^{(i)}$}\\
0, & \\text{otherwise}
\\end{cases}$$

-----

## 3\. Bias vs. Variance Diagnostics (Polynomial Regression)

By fitting a series of models with increasing complexity (e.g., increasing polynomial **degree**), the training and CV errors reveal whether the model suffers from high bias or high variance.

| Model Behavior | Training Error ($J_{\text{train}}$) | CV Error ($J_{cv}$) | Problem | Solution |
| :--- | :--- | :--- | :--- | :--- |
| **Low Degree** | High | High (approx. $J_{train} \approx J_{cv}$) | **High Bias** (Underfitting) | Increase model complexity (e.g., increase polynomial degree, add features/layers). |
| **High Degree** | Low | High (approx. $J_{train} \ll J_{cv}$) | **High Variance** (Overfitting) | Decrease model complexity (e.g., decrease polynomial degree, **add regularization**). |
| **Optimal** | Low | Low (approx. $J_{train} \approx J_{cv}$) | Ideal | None. |

### **Tuning Regularization $(\lambda)$**

For a complex model (e.g., high degree polynomial), varying the regularization parameter $\lambda$ controls the bias-variance trade-off:

  * **Low $\lambda$:** Leads to high variance (overfitting).
  * **High $\lambda$:** Leads to high bias (underfitting).
  * The **optimal $\lambda$** minimizes the **Cross-Validation Error** ($J_{cv}$).

### **Impact of Data Size ($m$)**

When a model has **High Variance (overfitting)**, acquiring and training on **more data** can significantly improve performance and generalization, causing $J_{\text{train}}$ and $J_{cv}$ to converge to a low value.

-----

## 4\. Neural Network Complexity and Regularization

The same principles apply to neural networks by varying the **architecture** (number of layers, number of units) or applying **regularization**.

### **Model Comparison (Multiclass Classification)**

Two neural network architectures were compared:

| Model | Architecture (Units/Layers) | Training Error ($J_{\text{train}}$) | CV Error ($J_{cv}$) | Diagnosis |
| :--- | :--- | :--- | :--- | :--- |
| **Complex** | 3 layers (120, 40, 6 units) | $\approx 0.003$ (Very Low) | $\approx 0.122$ (High) | **High Variance (Overfitting)** |
| **Simple** | 2 layers (6, 6 units) | $\approx 0.062$ (Higher) | $\approx 0.087$ (Lower) | Better balance, but higher overall error (possible **High Bias**) |
| **Regularized** | Complex + $\ell_2(\lambda=0.1)$ | $\approx 0.072$ (Higher) | $\approx 0.066$ (Lowest) | **Best Generalization** |

### **Regularization Implementation in Keras**

Regularization (specifically $\ell_2$ or Weight Decay) is implemented by adding `kernel_regularizer` to the `Dense` layers:

```python
# Example of the Complex Regularized Model
model_r = Sequential(
    [
        Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        Dense(40, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        Dense(6, activation='linear')
    ]
)
# Note: loss uses from_logits=True for numerical stability
model_r.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
)
```

The **Regularized** model, despite having a higher training error than the overfit complex model, achieved the lowest cross-validation error, demonstrating **improved generalization**.
