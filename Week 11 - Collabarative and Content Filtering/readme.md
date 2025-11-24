# 💡 Recommender Systems

Recommender Systems are tools that predict what a **user will like** (e.g., a movie, product, or song) to offer personalized options. The main goal is to estimate a **missing rating** ($\hat{r}_{i, j}$) for user $i$ and item $j$.

***

## 1. Collaborative Filtering (CF) 🤝

### The Core Idea: "Find people with similar hidden tastes."

CF works by analyzing **past user behavior (ratings)** across all items and users, completely ignoring the item's visible features (like its genre or color).

* **Intuition:** If you and another user rated items similarly in the past, you are likely to enjoy the same new items.

### Implementation: Matrix Factorization

The system learns two sets of **secret (hidden) feature vectors** simultaneously to explain all the known ratings in the huge User-Item matrix. This process is called **Matrix Factorization**.

| Key Element | Detail & Intuition | Formula |
| :--- | :--- | :--- |
| **User Vector** ($x^{(i)}$) | A hidden vector representing User $i$'s secret preferences (e.g., how much they like action vs. drama). | N/A |
| **Item Vector** ($\theta^{(j)}$) | A hidden vector representing Item $j$'s secret characteristics (e.g., how action-packed vs. dramatic the movie is). | N/A |
| **Prediction** ($\hat{r}_{i, j}$) | The predicted rating is the **multiplication** (dot product) of the User's secret preference vector and the Item's secret characteristic vector. | $$\hat{r}_{i,j} = x^{(i)} \cdot \theta^{(j)}$$ |
| **Learning Goal** (Cost Function $J$) | The model uses **Gradient Descent** to find the optimal $x^{(i)}$ and $\theta^{(j)}$ vectors that make the prediction error as small as possible. **Regularization** ($\lambda$) is added to prevent the vectors from becoming too complex (avoiding overfitting). | [$$J = \frac{1}{2} \sum_{(i,j): r(i,j)=1} ((\mathbf{x}^{(i)})^T\mathbf{\theta}^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{i, k} (\mathbf{x}_k^{(i)})^2 + \frac{\lambda}{2} \sum_{j, k} (\mathbf{\theta}_k^{(j)})^2$$] |

| **Pro** | **Con** |
| :--- | :--- |
| Finds **subtle, complex taste patterns**. | **Cold Start Problem:** Cannot recommend **brand new items** because they have no ratings data to learn their hidden vector ($\theta^{(j)}$). |

---

## 2. Content-Based Filtering (CBF) 📝

### The Approach: Using Item Tags and Attributes

CBF relies on the **known, visible features** (metadata/tags) of the items (like genre, director, or size).

* **Intuition:** "Since you liked items with these features in the past, you'll probably like another item that shares those features."

### Implementation: Neural Network (NN)

A small **Neural Network** is designed to map an item's features directly to a predicted rating for a specific user.

| Key Element | Detail & Intuition |
| :--- | :--- |
| **Input Layer** | Takes the **known features** of the item as a vector (e.g., a movie's genre tags and attributes). |
| **Model Structure** | The data flows through **Hidden Layers** (where complex, non-linear feature combinations are learned). |
| **Output Layer** | A single unit that directly outputs the **predicted rating** ($\hat{r}$) for that item by the target user. |
| **Training** | The model's weights are adjusted (using backpropagation in the NN) to match its output prediction to the user's actual past ratings. |

| **Pro** | **Con** |
| :--- | :--- |
| **Solves Cold Start for Items:** Can recommend **new items** instantly, as long as their features are tagged. | **Limited Discovery:** Tends to recommend items that are **too similar** to what the user already consumed. |

---

## 3. Real-World Scaling: The Retrieval & Ranking Funnel 🚀

In large systems (millions of items), calculating every prediction is too slow. Recommendations use a **two-step funnel** to ensure extreme speed and accuracy :

### Step 1: Retrieval (The Fast Filter)

* **Goal:** Quickly cut down millions of items to a small pool of highly relevant candidates (e.g., 100-500 items).
* **Intuition:** Use fast, basic rules or simple models to quickly filter the massive catalog.

| Example Retrieval Rules | How it Works (Conceptual Implementation) |
| :--- | :--- |
| **Last Seen Item Similarity** | Find items most similar to the **last movie the user rated highly** (e.g., retrieve items that are nearest neighbors to the last item's vector, $\theta^{(j)}$). |
| **Past Interaction Lookups** | Look up all items from the **same genre or director** as the user's top-rated movies. |
| **Popularity Filter** | Include items from the top 100 most **popular items** overall, if the user hasn't seen them. |

### Step 2: Ranking (The Deep Sort)

* **Goal:** Take the small list of candidates from Step 1 and **sort them precisely** from best to worst based on predicted relevance.
* **Intuition:** Now that we have a manageable list, we can afford to use the most complex and accurate models.
* **How:** Apply a sophisticated model (like a deep Neural Network or XGBoost) to score every item in the candidate list.
* **Output:** The final, personalized list of 10-20 recommendations presented to the user.

This funnel ensures **high accuracy** by using the complex models only on the small, retrieved set, delivering results almost instantly.
