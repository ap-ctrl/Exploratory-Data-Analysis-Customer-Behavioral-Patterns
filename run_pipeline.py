"""
End-to-End Execution Pipeline for RetailMind AI
Executes all stages sequentially:
  1. Data Ingestion & Quality Audit
  2. Cleaning, Cancellation Handling & Exploratory Data Analysis
  3. Feature Engineering & Customer Behavioral Segmentation
  4. Association Rule Mining (FP-Growth Global & Per-Segment)
  5. Recommender Instantiation & Offline Evaluation Benchmark
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

import pandas as pd
from src.config import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, SEGMENT_MIN_SUPPORT
from src.data_loader import convert_excel_to_parquet, generate_data_quality_metrics, build_data_quality_report, load_raw_data
from src.cleaning import run_cleaning_pipeline
from src.eda import run_behaviour_analysis
from src.features import create_time_split, compute_rfm_features
from src.segmentation import (
    preprocess_rfm, evaluate_kmeans_k, fit_final_kmeans,
    evaluate_stability, run_hierarchical, run_dbscan, profile_and_name_segments
)
from src.association import build_baskets, benchmark_fpgrowth_vs_apriori, mine_rules, save_rules_parquet, plot_association_outputs
from src.recommender import load_recommender
from src.evaluation import benchmark_recommenders_offline, evaluate_holdout_association_rules

def main():
    print("=" * 80)
    print("🛍️  RetailMind AI: End-to-End Data Mining & Recommendation Pipeline")
    print("=" * 80)

    # Stage 1: Data Ingestion & Quality Audit
    print("\n[Stage 1] Loading raw dataset and auditing data quality...")
    raw_df = convert_excel_to_parquet()
    quality_metrics = generate_data_quality_metrics(raw_df)
    build_data_quality_report(quality_metrics)

    # Stage 2: 7-Step Cleaning Pipeline & EDA
    print("\n[Stage 2] Running 7-step logged cleaning pipeline & customer behavior EDA...")
    cleaning_res = run_cleaning_pipeline(raw_df)
    df_all = cleaning_res["df_all"]
    df_cust = cleaning_res["df_cust"]
    cancellations = pd.read_parquet(PROCESSED_DIR / "cancellations.parquet")
    run_behaviour_analysis(df_all, df_cust, cancellations)

    # Stage 3: Temporal Train/Test Split & Customer Segmentation
    print("\n[Stage 3] Temporal split, RFM engineering, and clustering validation...")
    train_cust, test_cust, train_all, test_all = create_time_split(df_cust, df_all)
    rfm_train = compute_rfm_features(train_cust)
    X_scaled, scaler, _ = preprocess_rfm(rfm_train)

    evaluate_kmeans_k(X_scaled)
    evaluate_stability(X_scaled)
    run_hierarchical(X_scaled)
    run_dbscan(X_scaled)

    km, labels = fit_final_kmeans(X_scaled)
    summary, segment_map, df_segmented = profile_and_name_segments(rfm_train, labels, X_scaled)

    # Stage 4: Association Rule Mining via FP-Growth
    print("\n[Stage 4] Mining market basket affinity networks via FP-Growth...")
    basket_df, top_prods, desc_map = build_baskets(train_all)
    benchmark_fpgrowth_vs_apriori(basket_df)

    global_rules = mine_rules(basket_df, desc_map)
    save_rules_parquet(global_rules, MODELS_DIR / "rules_global.parquet")
    plot_association_outputs(global_rules)

    # Mine segment rules
    train_with_seg = train_cust.merge(df_segmented[["CustomerID", "segment_name"]], on="CustomerID", how="left")
    for seg_name, min_sup in SEGMENT_MIN_SUPPORT.items():
        sub_df = train_with_seg[train_with_seg["segment_name"] == seg_name]
        if len(sub_df) > 50:
            seg_baskets, _, _ = build_baskets(sub_df, min_basket_size=2)
            if not seg_baskets.empty:
                seg_rules = mine_rules(seg_baskets, desc_map, min_support=min_sup)
                slug = seg_name.lower().replace(" ", "_").replace("/", "_")
                save_rules_parquet(seg_rules, MODELS_DIR / f"rules_segment_{slug}.parquet")

    # Stage 5 & 6: Recommendation Engine & Offline Evaluation Benchmark
    print("\n[Stage 5 & 6] Instantiating Hybrid Recommender and running benchmark...")
    rec = load_recommender()
    holdout_eval = evaluate_holdout_association_rules(global_rules, test_cust)
    benchmark_df, user_eval_df = benchmark_recommenders_offline(rec, test_cust)

    print("\n" + "=" * 80)
    print("✅ Pipeline completed successfully! Ready to launch Streamlit:")
    print("   streamlit run app/streamlit_app.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
