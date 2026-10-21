import re
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
from src.config import PROCESSED_DIR, STOCK_CODE_REGEX

def step_drop_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Rule 1: Drop exact duplicate rows."""
    initial_len = len(df)
    df_clean = df.drop_duplicates()
    dropped = initial_len - len(df_clean)
    print(f"[Cleaning Step 1] Dropped exact duplicates: {dropped:,} rows removed. Remaining: {len(df_clean):,}")
    return df_clean, dropped

def step_extract_and_remove_cancellations(
    df: pd.DataFrame, 
    cancellations_path: Path = PROCESSED_DIR / "cancellations.parquet"
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Rule 2: Identify cancellations (InvoiceNo starts with 'C'), save separately, and remove from active sales."""
    initial_len = len(df)
    is_cancelled = df["InvoiceNo"].astype(str).str.startswith("C")
    df_cancellations = df[is_cancelled].copy()
    cancellations_path.parent.mkdir(parents=True, exist_ok=True)
    df_cancellations.to_parquet(cancellations_path, index=False, engine="pyarrow")
    
    df_clean = df[~is_cancelled].copy()
    dropped = initial_len - len(df_clean)
    print(f"[Cleaning Step 2] Extracted cancellations: {dropped:,} rows saved to {cancellations_path.name}. Remaining: {len(df_clean):,}")
    return df_clean, df_cancellations, dropped

def step_filter_positive_quantities_and_prices(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Rule 3: Keep only transactions where Quantity > 0 and UnitPrice > 0."""
    initial_len = len(df)
    mask = (df["Quantity"] > 0) & (df["UnitPrice"] > 0)
    df_clean = df[mask].copy()
    dropped = initial_len - len(df_clean)
    print(f"[Cleaning Step 3] Dropped non-positive Quantity/UnitPrice: {dropped:,} rows removed. Remaining: {len(df_clean):,}")
    return df_clean, dropped

def step_filter_real_products(df: pd.DataFrame, regex_pattern: str = STOCK_CODE_REGEX) -> Tuple[pd.DataFrame, int]:
    """Rule 4: Keep only real merchandise products matching 5 digits followed by optional letters."""
    initial_len = len(df)
    mask = df["StockCode"].astype(str).str.contains(regex_pattern, regex=True)
    df_clean = df[mask].copy()
    dropped = initial_len - len(df_clean)
    print(f"[Cleaning Step 4] Dropped administrative non-product StockCodes: {dropped:,} rows removed. Remaining: {len(df_clean):,}")
    return df_clean, dropped

def step_canonicalize_descriptions(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, Dict[str, str]]:
    """
    Rule 5: Build a canonical description per StockCode (most frequent non-null description, stripped and uppercase).
    Replace all descriptions with canonical version. Drop rows whose StockCode has no valid description.
    """
    initial_len = len(df)
    valid_desc_df = df[df["Description"].notnull() & (df["Description"].astype(str).str.strip() != "")].copy()
    valid_desc_df["clean_desc"] = valid_desc_df["Description"].astype(str).str.strip().str.upper()
    
    # Calculate mode (most frequent) description per StockCode
    canonical_map = (
        valid_desc_df.groupby("StockCode")["clean_desc"]
        .agg(lambda x: x.mode().iloc[0] if not x.empty else None)
        .dropna()
        .to_dict()
    )
    
    df_clean = df.copy()
    df_clean["Description"] = df_clean["StockCode"].map(canonical_map)
    # Drop rows without canonical description
    has_desc = df_clean["Description"].notnull()
    df_clean = df_clean[has_desc].copy()
    dropped = initial_len - len(df_clean)
    print(f"[Cleaning Step 5] Canonicalized descriptions across {len(canonical_map):,} products. Dropped {dropped:,} rows without valid descriptions. Remaining: {len(df_clean):,}")
    return df_clean, dropped, canonical_map

def step_add_datetime_and_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rule 6: Parse InvoiceDate and create TotalPrice, InvoiceDay, Month, DayOfWeek, Hour."""
    df_clean = df.copy()
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
    df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["UnitPrice"]
    df_clean["InvoiceDay"] = df_clean["InvoiceDate"].dt.date
    df_clean["Month"] = df_clean["InvoiceDate"].dt.to_period("M").astype(str)
    df_clean["DayOfWeek"] = df_clean["InvoiceDate"].dt.day_name()
    df_clean["Hour"] = df_clean["InvoiceDate"].dt.hour
    print(f"[Cleaning Step 6] Created TotalPrice, InvoiceDay, Month, DayOfWeek, and Hour features.")
    return df_clean

def step_create_datasets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rule 7: Split into df_all (including guests for global association) and df_cust (customers only for RFM).
    """
    df_all = df.copy()
    df_cust = df[df["CustomerID"].notnull()].copy()
    df_cust["CustomerID"] = df_cust["CustomerID"].astype(int)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(PROCESSED_DIR / "df_all.parquet", index=False, engine="pyarrow")
    df_cust.to_parquet(PROCESSED_DIR / "df_cust.parquet", index=False, engine="pyarrow")
    
    print(f"[Cleaning Step 7] Saved df_all.parquet ({len(df_all):,} rows) and df_cust.parquet ({len(df_cust):,} rows).")
    return df_all, df_cust

def run_cleaning_pipeline(raw_df: pd.DataFrame) -> Dict[str, Any]:
    """Executes the complete end-to-end cleaning workflow and returns metadata and datasets."""
    print("--- Starting End-to-End Data Cleaning Pipeline ---")
    df1, d1 = step_drop_duplicates(raw_df)
    df2, df_cancellations, d2 = step_extract_and_remove_cancellations(df1)
    df3, d3 = step_filter_positive_quantities_and_prices(df2)
    df4, d4 = step_filter_real_products(df3)
    df5, d5, canonical_map = step_canonicalize_descriptions(df4)
    df6 = step_add_datetime_and_financial_features(df5)
    df_all, df_cust = step_create_datasets(df6)
    
    summary = {
        "initial_rows": len(raw_df),
        "dropped_duplicates": d1,
        "cancellations_saved": d2,
        "dropped_non_positive_qty_price": d3,
        "dropped_non_products": d4,
        "dropped_no_description": d5,
        "final_df_all_rows": len(df_all),
        "final_df_cust_rows": len(df_cust),
        "canonical_products_count": len(canonical_map),
        "unique_customers_cleaned": int(df_cust["CustomerID"].nunique()),
        "total_gross_revenue_cleaned": float(df_all["TotalPrice"].sum())
    }
    print("--- Data Cleaning Pipeline Successfully Finished ---")
    return {"df_all": df_all, "df_cust": df_cust, "summary": summary}
