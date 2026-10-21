import pandas as pd
from pathlib import Path
from typing import Dict, Any
from src.config import RAW_EXCEL_PATH, RAW_PARQUET_PATH, REPORTS_DIR

def convert_excel_to_parquet(excel_path: Path = RAW_EXCEL_PATH, parquet_path: Path = RAW_PARQUET_PATH) -> pd.DataFrame:
    """Reads Excel file with specified dtypes and caches to parquet."""
    print(f"Reading {excel_path}...")
    dtype_dict = {"InvoiceNo": str, "StockCode": str}
    df = pd.read_excel(excel_path, dtype=dtype_dict, engine="openpyxl")
    # Clean mixed types in string columns for Arrow compatibility
    for col in ["InvoiceNo", "StockCode", "Description", "Country"]:
        if col in df.columns:
            # Preserve actual nulls while ensuring non-nulls are strings
            df[col] = df[col].astype("string")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    print(f"Saved raw parquet to {parquet_path} ({len(df):,} rows).")
    return df

def load_raw_data(parquet_path: Path = RAW_PARQUET_PATH) -> pd.DataFrame:
    """Loads raw data from parquet if available, otherwise converts from Excel first."""
    if not parquet_path.exists():
        return convert_excel_to_parquet(RAW_EXCEL_PATH, parquet_path)
    return pd.read_parquet(parquet_path, engine="pyarrow")

def generate_data_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes comprehensive data quality statistics."""
    total_rows = len(df)
    total_cols = df.shape[1]
    
    missing_counts = df.isnull().sum().to_dict()
    missing_pcts = (df.isnull().mean() * 100).round(2).to_dict()
    
    is_cancelled = df["InvoiceNo"].astype(str).str.startswith("C")
    cancellations_count = int(is_cancelled.sum())
    cancellations_pct = round(cancellations_count / total_rows * 100, 2)
    
    neg_zero_qty = int((df["Quantity"] <= 0).sum())
    neg_zero_qty_pct = round(neg_zero_qty / total_rows * 100, 2)
    
    neg_zero_price = int((df["UnitPrice"] <= 0).sum())
    neg_zero_price_pct = round(neg_zero_price / total_rows * 100, 2)
    
    exact_duplicates = int(df.duplicated().sum())
    exact_duplicates_pct = round(exact_duplicates / total_rows * 100, 2)
    
    # Non-product stock codes: codes not matching 5 digits + optional letters
    non_product_mask = ~df["StockCode"].astype(str).str.contains(r"^\d{5}[A-Za-z]*$", regex=True)
    non_product_series = df.loc[non_product_mask, "StockCode"].value_counts()
    non_product_codes = non_product_series.to_dict()
    
    metrics = {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "date_min": str(df["InvoiceDate"].min()),
        "date_max": str(df["InvoiceDate"].max()),
        "unique_invoices": int(df["InvoiceNo"].nunique()),
        "unique_stock_codes": int(df["StockCode"].nunique()),
        "unique_descriptions": int(df["Description"].dropna().nunique()),
        "unique_customers": int(df["CustomerID"].dropna().nunique()),
        "unique_countries": int(df["Country"].nunique()),
        "missing_counts": missing_counts,
        "missing_pcts": missing_pcts,
        "cancellations_count": cancellations_count,
        "cancellations_pct": cancellations_pct,
        "neg_zero_qty": neg_zero_qty,
        "neg_zero_qty_pct": neg_zero_qty_pct,
        "neg_zero_price": neg_zero_price,
        "neg_zero_price_pct": neg_zero_price_pct,
        "exact_duplicates": exact_duplicates,
        "exact_duplicates_pct": exact_duplicates_pct,
        "non_product_codes": non_product_codes,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    return metrics

def build_data_quality_report(metrics: Dict[str, Any], report_path: Path = REPORTS_DIR / "data_quality_report.md") -> Path:
    """Generates reports/data_quality_report.md from metrics."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build markdown table for missing values
    missing_table_rows = []
    for col in metrics["missing_counts"]:
        cnt = metrics["missing_counts"][col]
        pct = metrics["missing_pcts"][col]
        dtype = metrics["dtypes"][col]
        missing_table_rows.append(f"| `{col}` | `{dtype}` | {cnt:,} | {pct}% |")
    missing_table = "\n".join(missing_table_rows)
    
    # Build top non-product codes table
    top_non_products = list(metrics["non_product_codes"].items())[:10]
    non_prod_table_rows = [f"| `{code}` | {count:,} |" for code, count in top_non_products]
    non_prod_table = "\n".join(non_prod_table_rows)

    content = f"""# Stage 1 Data Quality and Understanding Report

## 1. Dataset Identification and Semantic Scope
- **Source:** UCI Machine Learning Repository — *Online Retail* (Daqing Chen et al.)
- **Semantic Definition:** **One row represents one line item of an invoice.** An individual shopping basket (order) corresponds to a unique `InvoiceNo` and may contain multiple line items.
- **Temporal Coverage:** `{metrics['date_min']}` to `{metrics['date_max']}` (~1 year, 1 week).
- **Core Dimensions:**
  - Total records (line items): **{metrics['total_rows']:,}**
  - Features: **{metrics['total_cols']}**
  - Unique Invoices: **{metrics['unique_invoices']:,}**
  - Unique Products (StockCodes): **{metrics['unique_stock_codes']:,}**
  - Unique Registered Customers: **{metrics['unique_customers']:,}**
  - Unique Countries: **{metrics['unique_countries']}**

---

## 2. Missing Value Analysis

| Feature | Data Type | Missing Count | Missing Percentage |
| :--- | :--- | :--- | :--- |
{missing_table}

### Observations on Missing Values:
1. **CustomerID (24.93% missing):** Exactly {metrics['missing_counts']['CustomerID']:,} rows lack a customer identifier. These correspond to guest checkouts, telephone orders, or unauthenticated online transactions. They are preserved for transaction-level analysis (global association rules and storewide popularity) but excluded for customer-level segmentation (RFM) and personalized evaluation.
2. **Description (0.27% missing):** Exactly {metrics['missing_counts']['Description']:,} rows lack product descriptions, often associated with inventory reconciliation or damaged stock.

---

## 3. Data Integrity, Cancellations, and Anomalies

| Category | Count | Percentage of Total Rows |
| :--- | :--- | :--- |
| **Exact Duplicate Rows** | {metrics['exact_duplicates']:,} | {metrics['exact_duplicates_pct']}% |
| **Cancellations (`InvoiceNo` starts with 'C')** | {metrics['cancellations_count']:,} | {metrics['cancellations_pct']}% |
| **Non-positive Quantities (`Quantity <= 0`)** | {metrics['neg_zero_qty']:,} | {metrics['neg_zero_qty_pct']}% |
| **Non-positive Unit Prices (`UnitPrice <= 0`)** | {metrics['neg_zero_price']:,} | {metrics['neg_zero_price_pct']}% |

### Non-Product StockCodes Found:
The dataset contains non-merchandise administrative codes that do not represent physical gift items:
| StockCode | Occurrence Count | Description / Role |
| :--- | :--- | :--- |
{non_prod_table}
*Additional codes include `AMAZONFEE`, `BANK CHARGES`, `CRUK` (Cancer Research UK), and `S` (samples).*

---

## 4. Dataset Scope and Real-World Limitations
1. **Single Retailer Bias:** Data reflects a single UK-based giftware retailer with heavy seasonal wholesale spikes (notably Q4 Christmas demand).
2. **Wholesale Concentration:** Many transactions involve bulk quantities (up to tens of thousands of units per order) alongside single-item retail purchases, leading to heavy-tailed spend distributions.
3. **No Customer Demographics:** No demographic data (age, gender, income) or customer locations (beyond national IP/country) exist. Customer modeling must rely strictly on behavioral click/purchase telemetry (RFM).
4. **Guest Checkouts:** Approximately one quarter of transactions cannot be attributed to recurring user accounts.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Data quality report saved to {report_path}.")
    return report_path
