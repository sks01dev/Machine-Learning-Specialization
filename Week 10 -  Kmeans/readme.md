## 🎨 K-Means Clustering: Simple Revision Notes

K-Means is a way to automatically find **$K$ groups (clusters)** in your unlabeled data. Its goal is to make the groups as **tight and distinct** as possible.

***

## 📌 The Big Idea

* **What it does:** Sorts similar data points into $K$ containers.
* **The Goal:** Minimize the **distance** between each data point and the center of its assigned group.
* **Key Concept:** The center of a group is called a **centroid**.

***

## ⚙️ How the Algorithm Works (3 Simple Steps)

The process repeats until the centers stop moving:

1.  **Start with Random Centers:** Pick **$K$ random points** in your data to be the initial group centers (centroids).
2.  **Assignment Step ("The Closest Rule"):** Assign every single data point to the **closest center**.
3.  **Update Step ("The New Average"):** Move each center to the **true average position** of all the points currently assigned to it.

| Key Function | What it does |
| :--- | :--- |
| `find_closest_centroids` | Does Step 2: Assigns each point to the nearest center. |
| `compute_centroid_means` | Does Step 3: Calculates the new average position for the center. |

***

## 🖼️ Application: Image Compression

K-Means is used to reduce the number of colors in a picture, which saves storage space.

1.  **Data = Pixels:** An image is just a massive collection of pixels. Each pixel is treated as a data point defined by its **Red, Green, and Blue (RGB) intensity values** (these are the three features).
2.  **Run K-Means:** You choose a small $K$ (e.g., $K=16$). K-Means finds the **$K$ most representative colors** (the best 16 colors) in the whole image.
3.  **Compress:** Every pixel's original color is **replaced** by the color of its closest representative center.
4.  **Result:** The image looks nearly the same, but now only uses $K$ colors total, leading to compression (saving storage space).

