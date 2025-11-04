
# Neural Networks for Multiclass Digit Recognition — Implementation Summary

This document summarizes the steps taken to implement a **multiclass classification** neural network (for digits 0-9) using **TensorFlow/Keras** in the `C2_W2_Assignment.ipynb` notebook.

-----

## 1\. Core Concepts Introduced

### **ReLU Activation**

  * **Formula:** $a = \max(0, z)$
  * **Use:** Replaces the sigmoid in hidden layers for continuous, non-linear relationships. It introduces an 'off' range where the output is zero, which is necessary for non-linearity in multi-unit networks.

### **Softmax Function**

  * **Formula:** $$a_j = \frac{e^{z_j}}{ \sum_{k=0}^{N-1}{e^{z_k} }}$$
  * **Use:** Converts the final linear layer's output vector $\mathbf{z}$ into a **probability distribution** $\mathbf{a}$, where all elements are between 0 and 1 and sum to 1. This is essential for multiclass output interpretation.

-----

## 2\. Dataset Preparation

| Feature | Description |
| :--- | :--- |
| **Problem** | Recognize 10 handwritten digits (0-9). |
| **Dataset Size** | 5,000 training examples (subset of MNIST). |
| **Input ($X$) Shape** | **(5000, 400)**. Each example is a $20 \times 20$ grayscale image unrolled into a 400-dimensional vector. |
| **Output ($y$) Shape** | **(5000, 1)**. Contains the digit label (0 to 9) for each example. |

-----

## 3\. Neural Network Architecture (Keras `Sequential` Model)

The model is a three-layer dense neural network: **400 inputs $\rightarrow$ 25 units $\rightarrow$ 15 units $\rightarrow$ 10 outputs**.

| Layer (Keras `Dense`) | Units ($s_{out}$) | Activation | Parameter Shapes (W, b) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Input Layer** | N/A | N/A | N/A | Uses `tf.keras.Input(shape=(400,))` to define input size. |
| **Layer 1 (L1)** | **25** | `relu` | W1: (400, 25), b1: (25,) | First hidden layer. |
| **Layer 2 (L2)** | **15** | `relu` | W2: (25, 15), b2: (15,) | Second hidden layer. |
| **Layer 3 (L3)** | **10** | `linear` | W3: (15, 10), b3: (10,) | Output layer (10 units for 10 classes). |

### **Keras Model Construction**

The model was defined using the `Sequential` API:

```python
tf.random.set_seed(1234)
model = Sequential(
    [
        tf.keras.Input(shape=(400,)),
        Dense(25, activation='relu', name="L1"),
        Dense(15, activation='relu', name="L2"),
        Dense(10, activation='linear', name="L3"), # Linear output for numerical stability
    ], name = "my_model"
)
```

-----

## 4\. Model Training

### **Softmax Placement Note**

For improved numerical stability during training, the **softmax activation is OMITTED from the final `Dense` layer (using `linear` activation)** and is instead **grouped with the loss function** by setting `from_logits=True` in `model.compile()`.

### **Compilation and Fit**

1.  **Compile:** Define the loss function and optimizer.
      * **Loss:** `tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)` is used, where the target $y$ is the expected digit (0-9).
      * **Optimizer:** `tf.keras.optimizers.Adam(learning_rate=0.001)` (a common choice).
    <!-- end list -->
    ```python
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )
    ```
2.  **Fit (Train):** Train the model on the data.
      * **Epochs:** Set to `40`.
      * **Batches:** Uses the default batch size (32), resulting in $\approx 157$ batches per epoch ($5000 / 32 \approx 156.25$).
    <!-- end list -->
    ```python
    history = model.fit(X, y, epochs=40)
    ```

-----

## 5\. Prediction

To get the final prediction, the trained model's output (logits) must be processed:

1.  **Get Logits:** Call `model.predict()` on the input data.
    $$\mathbf{z} = \text{model.predict}(\mathbf{x})$$
2.  **Apply Softmax:** Apply `tf.nn.softmax()` to convert the logits $\mathbf{z}$ into probabilities $\mathbf{a}$.
    $$\mathbf{a} = \text{softmax}(\mathbf{z})$$
3.  **Find Predicted Class:** Use `np.argmax()` to find the index of the highest probability, which is the predicted digit.
    $$\hat{y} = \text{argmax}(\mathbf{a})$$

This process successfully classifies the handwritten digits.
