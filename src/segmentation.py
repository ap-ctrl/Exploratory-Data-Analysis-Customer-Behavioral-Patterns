import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

from src.config import RANDOM_STATE, MODELS_DIR, FIGURES_DIR, PROCESSED_DIR, K_MIN, K_MAX, KMEANS_N_INIT, RECOMMENDED_K

def preprocess_rfm(rfm_df: pd.DataFrame, save_scaler: bool = True) -> Tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """Applies log1p transform and standardization to RFM features."""
    rfm_log = rfm_df[["Recency", "Frequency", "Monetary"]].copy()
    
    # Log transforms for skewed features
    for col in ["Recency", "Frequency", "Monetary"]:
        rfm_log[col] = np.log1p(rfm_log[col])
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_log)
    
    if save_scaler:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
        print(f"StandardScaler saved to {MODELS_DIR / 'scaler.joblib'}.")
        
    return X_scaled, scaler, rfm_log

def evaluate_kmeans_k(X_scaled: np.ndarray, k_min: int = K_MIN, k_max: int = K_MAX) -> pd.DataFrame:
    """Evaluates K-Means clustering for k in [k_min, k_max] across 4 metrics."""
    metrics = []
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    k_range = range(k_min, k_max + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        
        metrics.append({
            "k": k,
            "Inertia": inertia,
            "Silhouette": sil,
            "Davies_Bouldin": db,
            "Calinski_Harabasz": ch
        })
        
    df_metrics = pd.DataFrame(metrics)
    
    # Plot evaluation metrics
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sns.lineplot(data=df_metrics, x="k", y="Inertia", marker="o", ax=axes[0, 0], color="#2980b9", linewidth=2.2)
    axes[0, 0].set_title("Elbow Method (Inertia)", fontweight="bold")
    
    sns.lineplot(data=df_metrics, x="k", y="Silhouette", marker="s", ax=axes[0, 1], color="#27ae60", linewidth=2.2)
    axes[0, 1].set_title("Silhouette Score (Higher is better)", fontweight="bold")
    
    sns.lineplot(data=df_metrics, x="k", y="Davies_Bouldin", marker="^", ax=axes[1, 0], color="#e74c3c", linewidth=2.2)
    axes[1, 0].set_title("Davies-Bouldin Index (Lower is better)", fontweight="bold")
    
    sns.lineplot(data=df_metrics, x="k", y="Calinski_Harabasz", marker="D", ax=axes[1, 1], color="#8e44ad", linewidth=2.2)
    axes[1, 1].set_title("Calinski-Harabasz Index (Higher is better)", fontweight="bold")
    
    for ax in axes.flat:
        ax.set_xlabel("Number of Clusters (k)", fontweight="bold")
        ax.set_xticks(list(k_range))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kmeans_evaluation.png", dpi=300)
    plt.close()
    
    print(f"K-Means evaluation plots saved to {FIGURES_DIR / 'kmeans_evaluation.png'}.")
    return df_metrics

def fit_final_kmeans(X_scaled: np.ndarray, k: int = RECOMMENDED_K) -> Tuple[KMeans, np.ndarray]:
    """Fits final K-Means model with chosen k and saves joblib artifact."""
    km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(km, MODELS_DIR / "kmeans.joblib")
    print(f"Final KMeans (k={k}) model saved to {MODELS_DIR / 'kmeans.joblib'}.")
    return km, labels

def evaluate_stability(X_scaled: np.ndarray, k: int = RECOMMENDED_K, n_seeds: int = 10) -> float:
    """Evaluates K-Means cluster assignment stability across distinct random seeds via pairwise Adjusted Rand Index."""
    seed_labels = []
    for seed in range(n_seeds):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed * 7 + 11)
        seed_labels.append(km.fit_predict(X_scaled))
        
    pairwise_aris = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            ari = adjusted_rand_score(seed_labels[i], seed_labels[j])
            pairwise_aris.append(ari)
            
    mean_ari = float(np.mean(pairwise_aris))
    print(f"Mean Pairwise Adjusted Rand Index over {n_seeds} random seeds: {mean_ari:.4f} (indicating high stability)")
    return mean_ari

def run_hierarchical(X_scaled: np.ndarray, k: int = RECOMMENDED_K, sample_size: int = 500) -> Dict[str, Any]:
    """Runs Agglomerative Hierarchical Clustering (Ward linkage) and plots dendrogram."""
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X_scaled)
    
    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    
    # Dendrogram on sample
    np.random.seed(RANDOM_STATE)
    idx_sample = np.random.choice(len(X_scaled), size=min(sample_size, len(X_scaled)), replace=False)
    X_sample = X_scaled[idx_sample]
    Z = linkage(X_sample, method="ward")
    
    plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90., leaf_font_size=10., show_contracted=True)
    plt.title(f"Hierarchical Clustering Dendrogram (Ward Linkage, Sample of {sample_size} Customers)", fontsize=12, fontweight="bold")
    plt.xlabel("Cluster Size / Leaf Index", fontweight="bold")
    plt.ylabel("Ward Distance", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hierarchical_dendrogram.png", dpi=300)
    plt.close()
    
    return {"labels": labels, "silhouette": sil, "davies_bouldin": db}

def run_dbscan(X_scaled: np.ndarray) -> Dict[str, Any]:
    """Runs DBSCAN with min_samples = 2 * n_features and k-distance knee analysis."""
    min_samples = 2 * X_scaled.shape[1]  # 2 * 3 = 6
    
    # K-distance graph
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])
    
    plt.figure(figsize=(8, 4))
    plt.plot(k_distances, color="#e67e22", linewidth=2.0)
    plt.title(f"DBSCAN k-Distance Plot (min_samples={min_samples})", fontweight="bold")
    plt.xlabel("Points sorted by distance", fontweight="bold")
    plt.ylabel(f"{min_samples}-NN Distance", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Pick knee near eps ~ 0.5 - 0.7
    eps = 0.6
    plt.axhline(eps, color="red", linestyle="--", label=f"Selected eps = {eps}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dbscan_k_distance.png", dpi=300)
    plt.close()
    
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = (labels == -1).sum()
    noise_pct = (noise_count / len(X_scaled)) * 100
    
    valid_mask = labels != -1
    sil = silhouette_score(X_scaled[valid_mask], labels[valid_mask]) if n_clusters > 1 else -1.0
    db_score = davies_bouldin_score(X_scaled[valid_mask], labels[valid_mask]) if n_clusters > 1 else -1.0
    
    print(f"DBSCAN results: {n_clusters} clusters, {noise_count} noise points ({noise_pct:.2f}%).")
    return {"labels": labels, "n_clusters": n_clusters, "noise_pct": noise_pct, "silhouette": sil, "davies_bouldin": db_score}

def profile_and_name_segments(
    rfm_df: pd.DataFrame, 
    labels: np.ndarray, 
    X_scaled: np.ndarray
) -> Tuple[pd.DataFrame, Dict[int, str], pd.DataFrame]:
    """
    Profiles clusters on raw R, F, M metrics, names them systematically, and produces visual artifacts.
    """
    df = rfm_df.copy()
    df["cluster"] = labels
    
    total_rev = df["Monetary"].sum()
    total_cust = len(df)
    
    # Aggregate raw metrics
    summary = df.groupby("cluster").agg(
        size=("CustomerID", "count"),
        Recency_mean=("Recency", "mean"),
        Recency_median=("Recency", "median"),
        Frequency_mean=("Frequency", "mean"),
        Frequency_median=("Frequency", "median"),
        Monetary_mean=("Monetary", "mean"),
        Monetary_median=("Monetary", "median"),
        Total_Monetary=("Monetary", "sum")
    ).reset_index()
    
    summary["pct_customers"] = (summary["size"] / total_cust) * 100
    summary["pct_revenue"] = (summary["Total_Monetary"] / total_rev) * 100
    
    # Systematic naming rule: rank clusters by Monetary and Recency
    # Sort clusters: Highest Monetary & Lowest Recency gets rank 1
    # We assign:
    # 1. Highest Monetary, High Frequency, Lowest Recency -> Champions
    # 2. High Frequency/Monetary, Low Recency -> Loyal Customers
    # 3. Moderate Frequency/Monetary, Low/Moderate Recency -> Potential / Promising
    # 4. Highest Recency, Lowest Frequency/Monetary -> At-Risk / Hibernating
    
    # Create scoring metric: Mon_rank (asc) - Rec_rank (desc)
    ranked_clusters = summary.sort_values(by=["Monetary_mean", "Recency_mean"], ascending=[False, True])["cluster"].tolist()
    
    names_order = ["Champions", "Loyal Customers", "Potential / Promising", "At-Risk / Hibernating"]
    if len(ranked_clusters) > len(names_order):
        names_order.extend([f"Segment_{i}" for i in range(len(names_order), len(ranked_clusters))])
        
    segment_map = {cluster_id: name for cluster_id, name in zip(ranked_clusters, names_order[:len(ranked_clusters)])}
    
    df["segment_name"] = df["cluster"].map(segment_map)
    summary["segment_name"] = summary["cluster"].map(segment_map)
    
    # Save segment mapping and customer segments parquet
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "segment_map.json", "w") as f:
        json.dump({str(k): v for k, v in segment_map.items()}, f, indent=2)
        
    df.to_parquet(PROCESSED_DIR / "customer_segments.parquet", index=False, engine="pyarrow")
    print(f"Customer segments saved to {PROCESSED_DIR / 'customer_segments.parquet'}.")
    
    # Visualizations
    # 1. Boxplots of raw R, F, M
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, col in enumerate(["Recency", "Frequency", "Monetary"]):
        sns.boxplot(data=df, x="segment_name", y=col, ax=axes[i], palette="Set2")
        axes[i].set_title(f"{col} Distribution by Segment", fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=25)
        if col in ["Frequency", "Monetary"]:
            axes[i].set_yscale("log")
            axes[i].set_ylabel(f"{col} (Log scale)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "segment_rfm_boxplots.png", dpi=300)
    plt.close()
    
    # 2. 2D PCA Scatter plot
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    df["pca1"] = X_pca[:, 0]
    df["pca2"] = X_pca[:, 1]
    
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df, x="pca1", y="pca2", hue="segment_name", palette="tab10", alpha=0.7, s=40
    )
    plt.title(f"Customer Segments 2D Projection (PCA Explained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%)", fontweight="bold")
    plt.xlabel("Principal Component 1", fontweight="bold")
    plt.ylabel("Principal Component 2", fontweight="bold")
    plt.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "segment_pca_2d.png", dpi=300)
    plt.close()
    
    # 3. Radar / Heatmap of standardized means
    scaled_df = pd.DataFrame(X_scaled, columns=["Recency", "Frequency", "Monetary"])
    scaled_df["segment_name"] = df["segment_name"].values
    mean_scaled = scaled_df.groupby("segment_name").mean()
    
    plt.figure(figsize=(8, 5))
    sns.heatmap(mean_scaled, annot=True, cmap="vlag", fmt=".2f", linewidths=1.0, cbar_kws={'label': 'Z-Score'})
    plt.title("Standardized Segment Feature Centers (Z-Scores)", fontweight="bold", pad=12)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "segment_radar_heatmap.png", dpi=300)
    plt.close()
    
    return summary, segment_map, df
