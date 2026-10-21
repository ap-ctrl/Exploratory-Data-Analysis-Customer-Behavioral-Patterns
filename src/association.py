import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple, List
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules

from src.config import (
    TOP_N_PRODUCTS, MIN_SUPPORT_GLOBAL, MIN_SUPPORT_FALLBACK, 
    MAX_LEN_ITEMSETS, MIN_LIFT, MIN_CONFIDENCE, 
    MODELS_DIR, FIGURES_DIR, PROCESSED_DIR
)

def build_baskets(
    df: pd.DataFrame, 
    top_n: int = TOP_N_PRODUCTS, 
    min_basket_size: int = 2
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """
    Constructs a one-hot boolean DataFrame of transactions.
    Filters to top N products and discards invoices with fewer than min_basket_size items.
    """
    # 1. Product mapping
    desc_map = (
        df.groupby("StockCode")["Description"]
        .agg(lambda x: x.mode().iloc[0] if not x.empty else "UNKNOWN")
        .to_dict()
    )
    
    # 2. Filter to top N products by invoice appearance
    top_products = df.groupby("StockCode")["InvoiceNo"].nunique().nlargest(top_n).index.tolist()
    df_filtered = df[df["StockCode"].isin(top_products)].copy()
    
    # 3. Group by InvoiceNo
    baskets_series = df_filtered.groupby("InvoiceNo")["StockCode"].apply(lambda s: sorted(list(set(s))))
    # Filter baskets with >= min_basket_size
    baskets_series = baskets_series[baskets_series.apply(len) >= min_basket_size]
    transactions = baskets_series.tolist()
    
    print(f"Extracted {len(transactions):,} multi-item baskets across {len(top_products):,} catalog items.")
    
    # 4. Sparse TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions, sparse=True)
    basket_df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    
    return basket_df, top_products, desc_map

def benchmark_fpgrowth_vs_apriori(basket_df: pd.DataFrame, sample_size: int = 2000) -> Dict[str, float]:
    """Demonstrates that FP-Growth yields identical itemsets to Apriori while exhibiting superior runtime."""
    sample = basket_df.iloc[:min(sample_size, len(basket_df))].copy()
    
    # Measure Apriori
    start_ap = time.time()
    itemsets_ap = apriori(sample, min_support=0.03, use_colnames=True, max_len=2)
    time_ap = time.time() - start_ap
    
    # Measure FP-Growth
    start_fp = time.time()
    itemsets_fp = fpgrowth(sample, min_support=0.03, use_colnames=True, max_len=2)
    time_fp = time.time() - start_fp
    
    print(f"[Benchmark on {len(sample)} baskets] Apriori: {time_ap:.4f}s ({len(itemsets_ap)} itemsets), FP-Growth: {time_fp:.4f}s ({len(itemsets_fp)} itemsets)")
    return {"time_apriori": time_ap, "time_fpgrowth": time_fp, "speedup": time_ap / max(time_fp, 0.0001)}

def mine_rules(
    basket_df: pd.DataFrame,
    desc_map: Dict[str, str],
    min_support: float = MIN_SUPPORT_GLOBAL,
    min_confidence: float = MIN_CONFIDENCE,
    min_lift: float = MIN_LIFT,
    max_len: int = MAX_LEN_ITEMSETS
) -> pd.DataFrame:
    """Mines frequent itemsets with FP-Growth and extracts high-confidence recommendation rules."""
    print(f"Mining frequent itemsets with min_support={min_support}...")
    frequent_itemsets = fpgrowth(basket_df, min_support=min_support, use_colnames=True, max_len=max_len)
    
    if len(frequent_itemsets) < 10 and min_support > MIN_SUPPORT_FALLBACK:
        print(f"Itemsets count ({len(frequent_itemsets)}) too small; lowering min_support to {MIN_SUPPORT_FALLBACK}...")
        min_support = MIN_SUPPORT_FALLBACK
        frequent_itemsets = fpgrowth(basket_df, min_support=min_support, use_colnames=True, max_len=max_len)
        
    print(f"Discovered {len(frequent_itemsets):,} frequent itemsets.")
    if frequent_itemsets.empty:
        return pd.DataFrame()
        
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    print(f"Generated {len(rules):,} raw association rules.")
    
    # Add length columns
    rules["ant_len"] = rules["antecedents"].apply(len)
    rules["con_len"] = rules["consequents"].apply(len)
    
    # Filter constraints
    mask = (
        (rules["confidence"] >= min_confidence) &
        (rules["lift"] >= min_lift) &
        (rules["ant_len"] <= 2) &
        (rules["con_len"] == 1)
    )
    filtered_rules = rules[mask].copy().sort_values(by=["lift", "confidence"], ascending=[False, False])
    print(f"Filtered to {len(filtered_rules):,} actionable rules (confidence >= {min_confidence}, lift >= {min_lift}).")
    
    if filtered_rules.empty:
        return pd.DataFrame()
        
    # Unpack frozensets into explicit strings for parquet serialization
    filtered_rules["antecedents_str"] = filtered_rules["antecedents"].apply(lambda s: ",".join(sorted(list(s))))
    filtered_rules["consequents_str"] = filtered_rules["consequents"].apply(lambda s: list(s)[0])
    
    # Add human readable descriptions
    filtered_rules["antecedent_names"] = filtered_rules["antecedents"].apply(
        lambda s: " + ".join([desc_map.get(x, x) for x in s])
    )
    filtered_rules["consequent_name"] = filtered_rules["consequents"].apply(
        lambda s: desc_map.get(list(s)[0], list(s)[0])
    )
    
    return filtered_rules

def save_rules_parquet(rules: pd.DataFrame, filepath: Path) -> None:
    """Serializes rules dataframe safely to parquet, converting frozensets to strings."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    save_df = rules.drop(columns=["antecedents", "consequents"], errors="ignore").copy()
    save_df.to_parquet(filepath, index=False, engine="pyarrow")
    print(f"Saved {len(save_df):,} association rules to {filepath.name}.")

def plot_association_outputs(rules: pd.DataFrame) -> None:
    """Generates visual artifacts for Stage 4: Support vs Confidence scatter and Top Co-occurrence Heatmap."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if rules.empty:
        return
        
    # 1. Scatter Plot
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        rules["support"], rules["confidence"], 
        c=rules["lift"], cmap="plasma", alpha=0.75, edgecolors="none", s=50
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Lift Metric", fontweight="bold")
    plt.title("Association Rules: Support vs. Confidence (Coloured by Lift)", fontsize=12, fontweight="bold")
    plt.xlabel("Rule Support", fontweight="bold")
    plt.ylabel("Rule Confidence", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "association_scatter.png", dpi=300)
    plt.close()
    
    # 2. Top Product Co-occurrence Heatmap (Pairwise single-item rules)
    pairwise = rules[rules["ant_len"] == 1].head(15).copy()
    if not pairwise.empty:
        pivot_table = pairwise.pivot_table(
            index="antecedent_names", columns="consequent_name", values="lift", fill_value=0
        )
        # Clean labels to max 35 chars
        pivot_table.index = [str(x)[:35] for x in pivot_table.index]
        pivot_table.columns = [str(x)[:35] for x in pivot_table.columns]
        
        plt.figure(figsize=(11, 8))
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Lift'})
        plt.title("Cross-Selling Affinity Heatmap (Top Product Pairs by Lift)", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Recommended Consequent Product", fontweight="bold")
        plt.ylabel("Purchased Antecedent Product", fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "association_top_pairs_heatmap.png", dpi=300)
        plt.close()
        
    print(f"Association plots saved to {FIGURES_DIR}.")
