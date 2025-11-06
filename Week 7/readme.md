# 🌳 Decision Tree Classifier from Scratch (C2, Week 4)

This document provides a detailed summary of the problem, dataset, core functions, and the final structure of the Decision Tree built from scratch using **Information Gain** principles.

---

## I. Problem Statement and Dataset

### **Objective**
The primary goal is to build a classifier that can distinguish between **edible ($\mathbf{1}$)** and **poisonous ($\mathbf{0}$)** mushrooms based on physical attributes.

### **Dataset Overview**
* **Total Examples ($m$):** 10
* **Target ($y$):** Binary ($\mathbf{1}$ for Edible, $\mathbf{0}$ for Poisonous). Initially, the dataset has an equal split (5 Edible, 5 Poisonous), resulting in a maximum starting **Entropy of $H=1.0$**.
* **Features:** Three categorical features, which were **One-Hot Encoded** into binary values for ease of implementation.

| Feature Name | Index | Value $\mathbf{1}$ | Value $\mathbf{0}$ |
| :--- | :--- | :--- | :--- |
| **Brown Cap** | 0 | Brown | Red |
| **Tapering Stalk Shape** | 1 | Tapering | Enlarging |
| **Solitary** | 2 | Yes (Solitary) | No |

---

## II. Core Functions: The Splitting Criterion

The Decision Tree is built recursively by finding the feature that provides the greatest reduction in impurity (**Information Gain**) at each node.

### **1. `compute_entropy(y)`**
Calculates the **uncertainty** or **impurity** of the class labels in a given subset of data.

* **Formula:** $$H(p_1) = -p_1 \log_2(p_1) - (1- p_1) \log_2(1- p_1)$$
    * $p_1$ is the fraction of edible examples ($\mathbf{1}$).
    * If $p_1=0$ or $p_1=1$, the node is pure, and $H$ is set to $\mathbf{0}$ to handle $\mathbf{0} \log(\mathbf{0})$.

### **2. `split_dataset(X, node_indices, feature)`**
A utility function that partitions the indices of the data at a node based on the selected binary feature.

* **Logic:**
    * If $\mathbf{X}[\text{index}][\text{feature}] = \mathbf{1} \rightarrow$ **Left Branch** (`left_indices`).
    * If $\mathbf{X}[\text{index}][\text{feature}] = \mathbf{0} \rightarrow$ **Right Branch** (`right_indices`).

### **3. `compute_information_gain(...)`**
Calculates the effectiveness of a potential split by measuring the reduction in entropy.

* **Formula:** $$\text{Information Gain} = H(p_1^{\text{node}}) - (w^{\text{left}}H(p_1^{\text{left}}) + w^{\text{right}}H(p_1^{\text{right}}))$$
    * $H(p_1^{\text{node}})$: Entropy before the split.
    * $w^{\text{left/right}}$: The proportion of samples entering the left/right branch.

| Feature Split | Root Entropy | Weighted Child Entropy | Information Gain (IG) |
| :--- | :--- | :--- | :--- |
| **Brown Cap (0)** | $1.0$ | $\approx 0.965$ | $\approx 0.035$ |
| **Tapering Stalk (1)** | $1.0$ | $\approx 0.875$ | $\approx 0.125$ |
| **Solitary (2)** | $1.0$ | $\approx 0.722$ | $\approx \mathbf{0.278}$ |

### **4. `get_best_split(...)`**
Iterates over all available features, uses `compute_information_gain()`, and selects the feature that yields the **maximum IG**.

* **Result:** The best feature to split on at the root node is **Feature 2 (Solitary)**, as it has the highest Information Gain ($\approx 0.278$).

---

## III. Final Decision Tree Structure (Max Depth 2)

The recursive tree-building process stops when the `max_depth` of 2 is reached, resulting in the following structure:

### **Level 0: Root Node**
* **Split on:** **Feature 2 (Solitary)**.

### **Level 1: Left Branch (Solitary = $\mathbf{1}$) **
* *Impurity is reduced, but still mixed (4 Edible, 1 Poisonous).*
* **Split on:** **Feature 0 (Brown Cap)**.

### **Level 2: Final Leaves (Left Branch)**
| Split Condition | Leaf Indices | Class Distribution | Final Classification |
| :--- | :--- | :--- | :--- |
| **Left ($X_0=\mathbf{1}$) - Brown Cap** | [0, 1, 4, 7] | 4 Edible, 0 Poisonous | **Edible ($\mathbf{1}$)** (Pure Node) |
| **Right ($X_0=\mathbf{0}$) - Red Cap** | [5] | 0 Edible, 1 Poisonous | **Poisonous ($\mathbf{0}$)** (Pure Node) |

### **Level 1: Right Branch (Solitary = $\mathbf{0}$) **
* *Impurity is reduced, but still mixed (1 Edible, 4 Poisonous).*
* **Split on:** **Feature 1 (Tapering Stalk Shape)**.

### **Level 2: Final Leaves (Right Branch)**
| Split Condition | Leaf Indices | Class Distribution | Final Classification |
| :--- | :--- | :--- | :--- |
| **Left ($X_1=\mathbf{1}$) - Tapering** | [8] | 1 Edible, 0 Poisonous | **Edible ($\mathbf{1}$)** (Pure Node) |
| **Right ($X_1=\mathbf{0}$) - Enlarging** | [2, 3, 6, 9] | 0 Edible, 4 Poisonous | **Poisonous ($\mathbf{0}$)** (Pure Node) |

### So this completes the 2nd Course in the ML Specialization
