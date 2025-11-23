## 🎨 K-Means Clustering: Simple Revision Notes

K-Means is a way to automatically find **$K$ groups (clusters)** in your unlabeled data. Its goal is to make the groups as **tight and distinct** as possible.

* **What it does:** Sorts similar data points into $K$ containers.
* **The Goal:** Minimize the **distance** between each data point and the center of its assigned group.
* **Key Concept:** The center of a group is called a **centroid** ($\mu_k$).

***

## ⚙️ How the Algorithm Works (3 Steps + Formulas)

The process repeats until the centers stop moving:

1.  **Start with Random Centers:** Pick **$K$ random points** in your data to be the initial centroids ($\mu_k$).

2.  **Assignment Step ("The Closest Rule"):** Assign every point to the **closest center**.

    * **Formula (Finding Closest Center):** Find the cluster index $k$ that minimizes the squared distance between the data point $x^{(i)}$ and the centroid $\mu_k$.
        $$\text{index}(i) = \min_{k} ||x^{(i)} - \mu_k||^2$$

3.  **Update Step ("The New Average"):** Move each center to the **average position** of all points currently assigned to it.

    * **Formula (Updating the Center):** Calculate the new centroid $\mu_k$ as the mean of all points $x^{(i)}$ belonging to cluster $C_k$.
        $$\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x^{(i)}$$

***

## 🖼️ Application: Image Compression

K-Means is used to reduce the number of colors in a picture, which saves storage space.

1.  **Data = Pixels:** An image is a collection of pixels. Each pixel is a data point defined by its **Red, Green, and Blue (RGB) intensity values** (these are the three features).
2.  **Run K-Means:** You set a small $K$ (e.g., $K=16$) to find the **$K$ most representative colors** (centroids) in the image.
3.  **Compress:** Every pixel's original color is **replaced** by the color of its closest representative center.
4.  **Result:** The image looks nearly the same but now only uses $K$ colors total (Lossy Compression).

