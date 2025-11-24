# 🎬 Recommender Systems: Revision Guide

Recommender Systems are tools that predict what a **user will like** (a movie, product, or song) to help them quickly find relevant options.

***

## 🎯 The Core Prediction Goal

The fundamental problem is to estimate a **missing rating**—the score a user would give to an item they haven't interacted with yet.

The two main approaches, Collaborative and Content-Based, solve this by looking at different pieces of information.

***

## 🤝 1. Collaborative Filtering (CF)

### **The Concept: "People with similar taste liked this."**

CF ignores the specific content (like genre or color) and focuses entirely on **past user ratings and behavior**. It finds patterns based on how different users rate the same items.

### **How the Model Works (The Matrix Factorization Approach)**

The system learns two sets of vectors simultaneously:

1.  **User Preference Vector ($x^{(i)}$):** A hidden vector representing what User $i$ likes (e.g., they strongly prefer serious drama).
2.  **Item Feature Vector ($\theta^{(j)}$):** A hidden vector representing what Item $j$ is about (e.g., this movie is very serious and dramatic).

| Formula | Description (Jargon-Free) |
| :--- | :--- |
| **Prediction** ($\hat{r}_{i, j}$): $$\hat{r}_{i,j} = x^{(i)} \cdot \theta^{(j)}$$ | The predicted rating is simply the **multiplication** of the User's Preference vector and the Item's Feature vector. |
| **Cost Function** ($J$): | The system uses a large cost function that **punishes** the model for every wrong prediction (squared error), pushing it to find better $x^{(i)}$ and $\theta^{(j)}$ vectors. |

### **Key Strength & Weakness**
* **Strength:** Excellent at finding subtle, complex preference patterns that human tags might miss.
* **Weakness (The Cold Start Problem):** Cannot recommend **brand new items** because they have no ratings data, so the system can't calculate a $\theta^{(j)}$ vector.

***

## 📝 2. Content-Based Filtering (CBF)

### **The Concept: "You liked this item's features, so here is another item with the same features."**

CBF relies on **known, visible features** of the items (like genre, director, or size).

### **How the Model Works (The Neural Network Approach)**

A small **Neural Network** is trained to predict the rating based purely on the item's features.

* **Input Layer:** Takes the **known features** of the item (e.g., [1, 0, 0] for Genre: Action, Comedy, Drama).
* **Hidden Layers:** Learn complex combinations of those features.
* **Output Layer:** Directly outputs the **predicted rating** for that item by the user.
* **Training:** The model adjusts its weights (using techniques like **backpropagation**) to minimize the difference between its predicted rating and the actual user ratings.

### **Key Strength & Weakness**
* **Strength:** Solves the **Cold Start Problem** for items—it can recommend new items immediately, as long as their features are known.
* **Weakness:** Tends to recommend things that are **too similar** to past items, leading to less discovery.

***

## 🚀 Speed and Scalability: Retrieval and Ranking

In real-world applications (with millions of users and items), a system must work fast. Calculating every possible rating prediction is too slow.

This is solved using a **two-step funnel** :

### **1. Retrieval (The Fast Filter)**
* **Goal:** Quickly narrow down the millions of items to a **few hundred** or thousand strong candidates.
* **How:** Often done using simple CF methods, vector nearest neighbors searches, or quick content-based lookups (e.g., only show items the user hasn't seen from their favorite category).
* **Output:** A small set of relevant candidates (e.g., 500 items).

### **2. Ranking (The Deep Sort)**
* **Goal:** Take the small set of candidates and **sort them precisely** from best to worst.
* **How:** Done using a sophisticated, complex model (like the deep Neural Network or XGBoost) that takes more time but is run only on the small, retrieved set.
* **Output:** The final, highly accurate list of 10-20 recommendations presented to the user.
