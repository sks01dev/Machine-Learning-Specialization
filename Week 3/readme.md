# Logistic Regression — Quick Revision Notes  
**Machine Learning Specialization (C1 W3)** — *Andrew Ng*  

A compact summary of how to build **Logistic Regression** from scratch using **Python** and **NumPy**.  
Use these notes for quick revision and understanding of the model-building process.

---

## 📚 Table of Contents
1. [Concept Recap](#1️⃣-concept-recap)
2. [Implementation Steps](#2️⃣-implementation-steps)
   - [Step 1 — Import and Load Data](#🧩-step-1--import-and-load-data)
   - [Step 2 — Sigmoid Function](#⚙️-step-2--sigmoid-function)
   - [Step 3 — Cost Function](#📉-step-3--cost-function)
   - [Step 4 — Gradient Calculation](#🔁-step-4--gradient-calculation)
   - [Step 5 — Gradient Descent](#🚀-step-5--gradient-descent)
   - [Step 6 — Prediction](#🧮-step-6--prediction)
   - [Step 7 — Visualize Decision Boundary](#📊-step-7--visualize-decision-boundary)
3. [Key Intuitions](#3️⃣-key-intuitions)
4. [Optional — Regularization](#4️⃣-optional--regularization)
5. [Summary](#🏁-summary)

---

## 1️⃣ Concept Recap  

**Goal:** Predict a binary outcome (0 or 1).  
**Type:** Classification algorithm.  
**Idea:** Apply a linear function to input features and map it through a sigmoid to get a probability.  

$$
h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
$$  

The output represents the probability that \( y = 1 \) given \( x \).

---

## 2️⃣ Implementation Steps  

### 🧩 Step 1 — Import and Load Data  
```python
import numpy as np
import matplotlib.pyplot as plt
from utils import *

X_train, y_train = load_data("data/ex2data1.txt")
````

Visualize the dataset to see how the features separate the two classes.

---

### ⚙️ Step 2 — Sigmoid Function

The **sigmoid function** maps any real value to a range between 0 and 1.
It’s the core of logistic regression, ensuring outputs are interpretable as probabilities.

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

---

### 📉 Step 3 — Cost Function

Measures how well the model fits the data.
The **log loss (cross-entropy)** penalizes wrong confident predictions.

$$
J(\theta) = -\frac{1}{m}\sum \left[y\log(h_\theta(x)) + (1 - y)\log(1 - h_\theta(x))\right]
$$

Lower cost → better model fit.

---

### 🔁 Step 4 — Gradient Calculation

The gradient tells us how to update model parameters to minimize cost.

$$
\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m}\sum (h_\theta(x^{(i)}) - y^{(i)})
$$

These gradients are used in **gradient descent** to move toward optimal parameters.

---

### 🚀 Step 5 — Gradient Descent

Updates the parameters ( w ) and ( b ) iteratively to minimize cost.

$$
w := w - \alpha \frac{\partial J}{\partial w}, \quad
b := b - \alpha \frac{\partial J}{\partial b}
$$

Continue updating until convergence — when cost stops decreasing significantly.

---

### 🧮 Step 6 — Prediction

After training, use the learned parameters to predict outcomes on new data.

```python
def predict(X, w, b):
    probs = sigmoid(np.dot(X, w) + b)
    return probs >= 0.5  # Returns True/False (1/0)
```

The model outputs a probability → we classify as 1 if probability ≥ 0.5, otherwise 0.

---

### 📊 Step 7 — Visualize Decision Boundary

Plotting helps you see how the model separates the two classes.

```python
plot_decision_boundary(w, b, X_train, y_train)
```

If the boundary fits the data well, the model has learned meaningful parameters.

---

## 3️⃣ Key Intuitions

* Logistic regression predicts **probabilities**, not direct class labels.
* The **sigmoid** keeps predictions between 0 and 1.
* The **cost function (log loss)** heavily penalizes wrong confident predictions.
* **Gradient descent** optimizes parameters by minimizing cost.
* Works best for **linearly separable data** — when classes can be divided by a straight line.

---

## 4️⃣ Optional — Regularization

Used to prevent overfitting by penalizing large weights.

$$
J_{reg} = J + \frac{\lambda}{2m}\sum w_j^2
$$

Regularization makes the decision boundary smoother and improves generalization on unseen data.

---

## 🏁 Summary

### **Logistic Regression Workflow**

1. Import libraries and load data
2. Visualize the data
3. Implement the sigmoid function
4. Define the cost function
5. Compute gradients
6. Apply gradient descent to learn parameters
7. Make predictions
8. Plot the decision boundary

**Core Idea:**
Logistic regression models the probability of a binary outcome using a **sigmoid transformation** on a **linear function** of input features.
By optimizing its parameters through **gradient descent**, it learns the best separating boundary between two classes.

**End Goal:**
Understand the complete process of building, training, and evaluating a logistic regression model — the foundation for more advanced models like neural networks.


Would you like me to make this into a **`.md` file (README.md)** you can download directly?
```
