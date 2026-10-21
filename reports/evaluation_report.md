# Stage 6 Comprehensive Offline Evaluation Report

## 1. Customer Segmentation Model Evaluation
- **Method Chosen:** **K-Means Clustering ($k=4$)**
- **Clustering Validation Metrics:**
  - **Silhouette Score:** `0.3694`
  - **Davies-Bouldin Index:** `0.9411`
  - **Calinski-Harabasz Index:** `2889.7`
  - **Algorithmic Stability:** Evaluated over 10 random initialization seeds; achieved a **Mean Pairwise Adjusted Rand Index (ARI) of 0.9984**, confirming exceptional cluster boundary stability.
- **Comparison Against Alternative Unsupervised Paradigms:**
  - **Hierarchical (AGNES Ward Linkage, $k=4$):** Silhouette = `0.3280`, Davies-Bouldin = `1.0006`. Produced comparable clusters but higher computational complexity ($O(N^2)$).
  - **DBSCAN:** Discovered 3 core density clusters with **1.19% noise/outlier points**. Identified ultra-large wholesale accounts as density noise rather than cohesive segments, verifying that K-Means is better aligned with operational retail tiered segmentation.

---

## 2. Product Association Rule Mining & Hold-Out Validation
- **Global Rules Mined:** **2,342 actionable rules** (Filtered with $	ext{confidence} \ge 0.30$, $	ext{lift} \ge 1.50$, antecedent length $\le 2$).
- **Top Product Affinities:** Strongest lift affinities occur among complementary decorative gift sets (e.g., *Collage Paper Cups + Plates*, *Herb Markers*, and *Regency Teaware collection* with lift metrics exceeding 40–60).
- **Hold-Out Rule Generalizability on Test Baskets:**
  - Test-firing rules evaluated: **296 rules**.
  - **Mean Hold-out Confidence:** **66.11%**.
  - **Confidence Retention Rate:** **96.3%** of rules retained $\ge 50\%$ of their training confidence level in the unseen test period, confirming robust cross-temporal stability.

---

## 3. Personalized Recommendation System Offline Benchmark
- **Evaluation Users:** Exactly **1,838 customers** who made purchases in both the training window and the post-split test period (`2011-10-01` to `2011-12-09`).
- **Evaluation Protocol:** **Protocol B (New Items Only)** — Ground truth and candidate generation strictly exclude products previously purchased by the user, measuring genuine cross-selling discovery.

### Offline Benchmark Performance at $K=10$:
| Model                              |   Hit_Rate |   Precision |    Recall |      NDCG |   Catalog_Coverage_Pct | 95% CI (Hit-Rate@10)   |
|:-----------------------------------|-----------:|------------:|----------:|----------:|-----------------------:|:-----------------------|
| 1. Global Popularity               |   0.274755 |   0.0392818 | 0.0161922 | 0.041789  |                1.9703  | [0.2552, 0.2965]       |
| 2. Segment Popularity              |   0.301959 |   0.0429271 | 0.0192098 | 0.0466464 |                2.34152 | [0.2807, 0.3221]       |
| 3. Global Rules Only               |   0.348205 |   0.0581066 | 0.0291132 | 0.0691333 |                6.48201 | [0.3281, 0.3700]       |
| 4. Ablation (Hybrid w/o Seg Rules) |   0.394995 |   0.0652884 | 0.0342898 | 0.0756204 |                6.62479 | [0.3738, 0.4168]       |
| 5. Proposed (Hybrid Complete)      |   0.408596 |   0.068444  | 0.0381737 | 0.0781703 |               10.9937  | [0.3852, 0.4309]       |

---

## 4. Personalization and Segment-Level Breakdown
- **Catalog Personalization Diversity:** Average pairwise Jaccard similarity across customer top-10 recommendation lists was **0.0624** (low overlap confirms highly personalized, non-generic recommendations).
- **Performance by Customer Segment (Proposed Model @ $K=10$):**
| segment_name          |      hit |      ndcg |
|:----------------------|---------:|----------:|
| At-Risk / Hibernating | 0.378973 | 0.0741683 |
| Champions             | 0.390977 | 0.0659479 |
| Loyal Customers       | 0.426056 | 0.0784394 |
| Potential / Promising | 0.443769 | 0.102445  |

### Observations & Honest Discussion of Offline Results:
1. **Value of Segmentation & Rules:** Blending segment-specific association rules with popularity elevates Hit-Rate and NDCG over raw global popularity while significantly boosting catalog coverage from narrow bestsellers to diverse departmental giftware.
2. **Wholesale Purchase Characteristics:** In B2B wholesale giftware, repeat re-ordering of familiar SKUs is naturally prevalent. Under the strict "New Items Only" protocol, the hybrid system successfully uncovers novel cross-sell affinities (e.g., party accessories, kitchen tins) without relying on trivial re-purchases.
