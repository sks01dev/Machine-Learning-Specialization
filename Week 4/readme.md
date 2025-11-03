# Neural Networks for Handwritten Digit Recognition

This document summarizes the key concepts from the **C2\_W1\_Assignment.ipynb** notebook.
It focuses on **binary classification** of handwritten digits (0 and 1) using neural networks, implemented with both **NumPy** and **TensorFlow**.

-----

## Objective

Build and understand a simple neural network that classifies digits 0 and 1.
Learn the principles of **forward propagation**, **loss functions**, and **gradient descent**, and compare manual (NumPy) and automated (TensorFlow) implementations.

-----

## Core Concepts

### 1\. Neural Network Architecture

A neural network consists of:

  * Input layer
  * Hidden layer(s)
  * Output layer

Each layer computes:

$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = g(z^{[l]})$$

where $g$ is an **activation function** such as the sigmoid.

### 2\. Activation Function

**Sigmoid function:**

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Outputs values between 0 and 1, suitable for **binary classification**.

### 3\. Loss Function

**Binary Cross-Entropy (Log Loss):**

$$L = -\frac{1}{m}\sum [y\log(\hat{y}) + (1 - y)\log(1 - \hat{y})]$$

### 4\. Forward Propagation

Example using NumPy:

```python
Z1 = np.dot(W1, X) + b1
A1 = sigmoid(Z1)
Z2 = np.dot(W2, A1) + b2
A2 = sigmoid(Z2)
```

The output `A2` represents the predicted probability of class 1.

### 5\. Backpropagation (Conceptual)

Backpropagation computes **gradients** of the loss with respect to weights and biases:

  * $dW^{[l]} = \frac{dL}{dW^{[l]}}$
  * $db^{[l]} = \frac{dL}{db^{[l]}}$

Gradients are used to update parameters via gradient descent.

### 6\. Gradient Descent

Parameter update rule:

$$W := W - \alpha \cdot dW, \quad b := b - \alpha \cdot db$$

where $\alpha$ is the **learning rate**.

-----

## TensorFlow vs NumPy Implementation

| Concept | TensorFlow | NumPy |
| :--- | :--- | :--- |
| **Model Definition** | Uses `Sequential()` or `Dense` layers | Manual weight matrices |
| **Forward Propagation** | Built-in operations | Manual matrix math (`np.dot`) |
| **Training** | `model.fit()` | Manual gradient updates |
| **Activation & Loss** | Handled internally | Implemented manually |
| **Use Case** | Practical model training | Understanding fundamentals |

-----

## Vectorization

**Vectorization** replaces loops with **matrix operations** for faster computation.

Example:

```python
# Non-vectorized
for i in range(m):
    z[i] = np.dot(W, X[:, i]) + b

# Vectorized
Z = np.dot(W, X) + b
```

### Broadcasting

NumPy automatically expands arrays with compatible shapes:

```python
A = np.array([[1, 2, 3]])
B = np.array([[1], [2], [3]])
A + B  # Broadcasting adds row and column vectors
```

-----

## Key Takeaways

  * Neural networks learn **nonlinear relationships** between inputs and outputs.
  * Sigmoid activation outputs **probabilities** for binary outcomes.
  * Cross-entropy loss measures prediction accuracy.
  * Backpropagation applies the **chain rule** to compute gradients.
  * TensorFlow automates training, but NumPy reveals the **underlying mechanics**.
  * Vectorization and broadcasting improve **efficiency** and **readability**.

-----

## Formula Summary

| Concept | Formula |
| :--- | :--- |
| **Linear step** | $Z = W X + b$ |
| **Activation** | $A = g(Z)$ |
| **Sigmoid** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ |
| **Cost** | $J = -\frac{1}{m}\sum(y\log(\hat{y}) + (1-y)\log(1-\hat{y}))$ |
| **Gradient Descent** | $W := W - \alpha dW, \quad b := b - \alpha db$ |

-----

## Revision Checklist

  * Understand forward and backward propagation
  * Recall sigmoid and cost equations
  * Apply vectorization for efficiency
  * Compare TensorFlow and NumPy approaches
  * Recognize the role of the learning rate in convergence

-----

**End of Revision Notes**
This summary is intended for quick recall before tests, interviews, or revisiting neural network fundamentals.
