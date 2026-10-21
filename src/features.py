import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from src.config import PROCESSED_DIR, SPLIT_DATE

def create_time_split(
    df_cust: pd.DataFrame, 
    df_all: pd.DataFrame, 
    split_date: str = SPLIT_DATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits cleaned data strictly at split_date (default 2011-10-01) to prevent temporal data leakage.
    Saves train.parquet and test.parquet.
    """
    split_dt = pd.to_datetime(split_date)
    
    train_cust = df_cust[df_cust["InvoiceDate"] < split_dt].copy()
    test_cust = df_cust[df_cust["InvoiceDate"] >= split_dt].copy()
    
    train_all = df_all[df_all["InvoiceDate"] < split_dt].copy()
    test_all = df_all[df_all["InvoiceDate"] >= split_dt].copy()
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_cust.to_parquet(PROCESSED_DIR / "train.parquet", index=False, engine="pyarrow")
    test_cust.to_parquet(PROCESSED_DIR / "test.parquet", index=False, engine="pyarrow")
    train_all.to_parquet(PROCESSED_DIR / "train_all.parquet", index=False, engine="pyarrow")
    test_all.to_parquet(PROCESSED_DIR / "test_all.parquet", index=False, engine="pyarrow")
    
    print(f"[Time Split @ {split_date}]")
    print(f"  Train Cust rows: {len(train_cust):,} (Date range: {train_cust['InvoiceDate'].min()} to {train_cust['InvoiceDate'].max()})")
    print(f"  Test Cust rows:  {len(test_cust):,} (Date range: {test_cust['InvoiceDate'].min()} to {test_cust['InvoiceDate'].max()})")
    print(f"  Train All rows:  {len(train_all):,}, Test All rows: {len(test_all):,}")
    return train_cust, test_cust, train_all, test_all

def compute_rfm_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Core RFM metrics per customer using train data only.
    Recency: days between (max_train_date + 1 day) and customer's last invoice.
    Frequency: unique InvoiceNo count.
    Monetary: sum of TotalPrice.
    """
    reference_date = train_df["InvoiceDate"].max() + pd.Timedelta(days=1)
    
    rfm = train_df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda d: (reference_date - d.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum")
    ).reset_index()
    
    # Ensure all values positive
    rfm["Monetary"] = rfm["Monetary"].clip(lower=0.01)
    return rfm

def compute_extended_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes extended behavioral features for comparative clustering.
    - avg_basket_value: average monetary spend per invoice
    - avg_items_per_invoice: average total quantity per invoice
    - distinct_products: count of unique StockCodes purchased
    - avg_days_between_purchases: mean inter-purchase duration
    """
    rfm = compute_rfm_features(train_df)
    
    # Invoice-level aggregates
    inv_agg = train_df.groupby(["CustomerID", "InvoiceNo"]).agg(
        inv_spend=("TotalPrice", "sum"),
        inv_qty=("Quantity", "sum"),
        inv_date=("InvoiceDate", "min")
    ).reset_index()
    
    # Customer-level extended stats
    cust_ext = inv_agg.groupby("CustomerID").agg(
        avg_basket_value=("inv_spend", "mean"),
        avg_items_per_invoice=("inv_qty", "mean")
    ).reset_index()
    
    # Distinct products
    distinct_prod = train_df.groupby("CustomerID")["StockCode"].nunique().reset_index(name="distinct_products")
    
    # Inter-purchase interval calculation
    def calc_mean_interval(dates):
        sorted_d = dates.sort_values().drop_duplicates()
        if len(sorted_d) < 2:
            return 0.0
        diffs = sorted_d.diff().dropna().dt.total_seconds() / 86400.0
        return diffs.mean()
    
    intervals = inv_agg.groupby("CustomerID")["inv_date"].apply(calc_mean_interval).reset_index(name="avg_days_between_purchases")
    
    # Merge all
    extended_df = rfm.merge(cust_ext, on="CustomerID").merge(distinct_prod, on="CustomerID").merge(intervals, on="CustomerID")
    return extended_df
