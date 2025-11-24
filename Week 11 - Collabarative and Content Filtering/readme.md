# 💡 Recommender Systems

Recommender Systems are tools that predict what a **user will like** (e.g., a movie, product, or song) to offer personalized options.  
The main goal is to estimate a **missing rating** \(\hat{r}_{i,j}\) for user \(i\) and item \(j\).

---

## 1. Collaborative Filtering (CF) 🤝

### **Core Idea:** “Find people with similar hidden tastes.”

CF works by analyzing **past user behavior (ratings)** across all users and items, ignoring item features like genre or category.

**Intuition:**  
If two users rated items similarly before, they likely enjoy the same new items.

### **Matrix Factorization**

The system learns two **hidden feature vectors** to explain all known ratings in the User-Item matrix:

- **User vector**: \(x^{(i)}\) — represents user \(i\)'s hidden preferences  
- **Item vector**: \(\theta^{(j)}\) — represents item \(j\)'s hidden properties  

### **Prediction Formula**

\[
\hat{r}_{i,j} = x^{(i)} \cdot \theta^{(j)}
\]

### **Cost Function (Optimization Objective)**

\[
J = 
\frac{1}{2} 
\sum_{(i,j): r(i,j)=1}  
\left( (x^{(i)})^T \theta^{(j)} - y^{(i,j)} \right)^2 
+ \frac{\lambda}{2} \sum_{i,k} (x_k^{(i)})^2 
+ \frac{\lambda}{2} \sum_{j,k} (\theta_k^{(j)})^2
\]

### **Pros & Cons**

| **Pro** | **Con** |
|--------|---------|
| Finds **complex patterns** in user tastes. | **Cold Start Problem:** Cannot recommend **brand new items** (no ratings → no item vector). |

---

## 2. Content-Based Filtering (CBF) 📝

### **Approach:** Use item attributes or tags.

CBF relies on **known features** (metadata) such as genre, category, keywords, etc.

**Intuition:**  
“If you liked items with these features, you will like similar items.”

### **Neural Network Implementation**

| Component | Description |
|----------|-------------|
| **Input Layer** | Receives item’s known features. |
| **Hidden Layers** | Learn non-linear feature interactions. |
| **Output Layer** | Predicts the user’s rating \(\hat{r}\). |
| **Training** | Adjust weights via backprop using user’s past ratings. |

### **Pros & Cons**

| **Pro** | **Con** |
|--------|---------|
| Solves **item cold start** — new items can be recommended immediately. | Offers **limited discovery** (recommends too-similar items). |

---

## 3. Real-World Scaling: Retrieval & Ranking Funnel 🚀

Big systems (Netflix, YouTube, TikTok) use a **two-stage pipeline**:

---

## **Step 1: Retrieval (Fast Candidate Selection)**

Goal: Reduce millions of items → **~100–500 candidates**.

| Retrieval Strategy | Description |
|--------------------|-------------|
| **Last-Seen Item Similarity** | Find items nearest to the last highly-rated item’s vector \(\theta^{(j)}\). |
| **Genre / Feature Lookups** | Fetch items having features similar to user’s favorites. |
| **Popularity Filter** | Add globally popular items the user hasn’t consumed. |

---

## **Step 2: Ranking (Precise Sorting)**

Goal: Sort the retrieved candidates by predicted relevance.

- Uses a complex model (deep NN, XGBoost, etc.)
- Much slower but run on a **small item list**, so total latency stays low.
- Produces the final top **10–20 personalized recommendations**.

---

