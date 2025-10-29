# Linear Regression using Gradient Descent — Pipeline Overview

## 🎯 Objective
Learn parameters **w (weight)** and **b (bias)** that best fit a linear relationship between inputs `x` and outputs `y`, by minimizing the **Mean Squared Error (MSE)** cost function using **Gradient Descent**.

---

## 🚀 Step-by-Step Workflow

### **Step 1: Data Collection**
Gather or generate your training data — pairs of inputs and outputs `(x, y)`.

```python
# Example dataset
x_train = [1.0, 2.0, 3.0]
y_train = [2.0, 4.0, 6.0]
````

---

### **Step 2: Initialize Parameters**

Start with small or zero values for parameters.

```python
w = 0.0
b = 0.0
```

---

### **Step 3: Define the Model Function**

The linear model predicts output using:
[
$$\hat{y} = w x + b$$
]

```python
def compute_model_output(x, w, b):
    return w * x + b
```

---

### **Step 4: Define the Cost Function**

Use **Mean Squared Error (MSE)** to quantify how far predictions are from targets.

[
$$\J(w, b) = \frac{1}{2m}\sum_{i=1}^{m}\big(\hat{y}_i - y_i\big)^2
]

```python
def compute_cost(x, y, w, b):
    m = len(x)
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i])**2
    return cost / (2 * m)
```

---

### **Step 5: Compute Gradients**

Find the direction in which to adjust `w` and `b` to minimize cost.

[
$$\frac{\partial J}{\partial w} = \frac{1}{m}\sum_{i=1}^{m}\big(\hat{y}_i - y_i\big)\,x_i$$
$$\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}\big(\hat{y}_i - y_i\big)$$]
]

```python
def compute_gradient(x, y, w, b):
    m = len(x)
    dj_dw, dj_db = 0, 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db
```

---

### **Step 6: Gradient Descent Parameter Update**

Iteratively update parameters using the gradients.

[
$$w \leftarrow w - \alpha\,\frac{\partial J}{\partial w}$$
$$b \leftarrow b - \alpha\,\frac{\partial J}{\partial b}$$
]

```python
def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    w, b = w_in, b_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
    return w, b
```

---

### **Step 7: Train the Model**

Run gradient descent for a fixed number of iterations or until convergence.

```python
w_final, b_final = gradient_descent(x_train, y_train, w, b, alpha=0.01, num_iters=1000)
```

---

### **Step 8: Evaluate the Model**

Check the optimized parameters and final cost.

```python
print(f"Optimized parameters: w = {w_final}, b = {b_final}")
print(f"Final cost: {compute_cost(x_train, y_train, w_final, b_final)}")
```

---

### **Step 9: Inference / Prediction**

Use the trained model to make predictions on new inputs.

```python
x_new = 4.0
y_pred = compute_model_output(x_new, w_final, b_final)
print(f"Prediction for x={x_new}: y={y_pred}")
```

---

## 🧩 Summary of the Pipeline

| Step | Task                  | Description                     |
| ---- | --------------------- | ------------------------------- |
| 1    | Data Collection       | Gather input-output pairs       |
| 2    | Initialize Parameters | Start with initial `w` and `b`  |
| 3    | Define Model          | Linear equation ( y = wx + b )  |
| 4    | Compute Cost          | Measure prediction error (MSE)  |
| 5    | Compute Gradients     | Derive partial derivatives      |
| 6    | Update Parameters     | Apply gradient descent updates  |
| 7    | Train                 | Repeat until convergence        |
| 8    | Evaluate              | Assess performance and cost     |
| 9    | Predict               | Use trained model for inference |

---

## 🧠 Generalization to Future Models

This **pipeline skeleton** applies to all ML architectures — the only parts that change are:

| Component            | What Changes                  | Example                                                 |
| -------------------- | ----------------------------- | ------------------------------------------------------- |
| Model Function       | Structure of ( f(x) )         | Linear, Logistic, Neural Network, Transformer           |
| Cost Function        | Loss definition               | MSE, Cross-Entropy, Hinge Loss                          |
| Gradient Computation | Depends on model architecture | Manual (Linear), Backpropagation (NN, RNN, Transformer) |

The rest (data collection → parameter update → evaluation) remains identical across ML models.

---

## 📘 Summary

> **Core Idea:** Gradient Descent minimizes cost by iteratively adjusting parameters in the direction of steepest descent.
> **Key Goal:** Find `w` and `b` that minimize the cost function and produce the best fit line for your data.
