import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any
from src.config import FIGURES_DIR, REPORTS_DIR, PROCESSED_DIR

# Set aesthetic styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

def run_behaviour_analysis(df_all: pd.DataFrame, df_cust: pd.DataFrame, cancellations_df: pd.DataFrame) -> Dict[str, Any]:
    """Generates all Stage 2 behavior plots and statistical metrics."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Monthly revenue, invoices, and customer counts
    monthly_stats = df_cust.groupby("Month").agg(
        revenue=("TotalPrice", "sum"),
        invoices=("InvoiceNo", "nunique"),
        customers=("CustomerID", "nunique")
    ).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    
    sns.barplot(data=monthly_stats, x="Month", y="revenue", ax=ax1, color="#3498db", alpha=0.7)
    sns.lineplot(data=monthly_stats, x="Month", y="customers", ax=ax2, color="#e74c3c", marker="o", linewidth=2.5, label="Active Customers")
    sns.lineplot(data=monthly_stats, x="Month", y="invoices", ax=ax2, color="#2ecc71", marker="s", linewidth=2.5, label="Unique Invoices")
    
    ax1.set_title("Monthly Revenue, Invoices, and Active Customer Trajectory", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Month", fontweight="bold")
    ax1.set_ylabel("Revenue (£)", color="#2980b9", fontweight="bold")
    ax2.set_ylabel("Count", color="#27ae60", fontweight="bold")
    ax1.tick_params(axis="x", rotation=45)
    ax2.grid(False)
    ax2.legend(loc="upper left")
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "monthly_revenue_and_counts.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    
    # 2. Revenue by day of week and hour
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]  # Note: Saturday has no retail activity
    dow_stats = df_all.groupby("DayOfWeek")["TotalPrice"].sum().reindex([d for d in day_order if d in df_all["DayOfWeek"].unique()])
    hour_stats = df_all.groupby("Hour")["TotalPrice"].sum()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x=dow_stats.index, y=dow_stats.values, ax=ax1, palette="Blues_r")
    ax1.set_title("Total Revenue by Day of Week", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Day of Week", fontweight="bold")
    ax1.set_ylabel("Revenue (£)", fontweight="bold")
    ax1.tick_params(axis="x", rotation=30)
    
    sns.barplot(x=hour_stats.index, y=hour_stats.values, ax=ax2, palette="viridis")
    ax2.set_title("Total Revenue by Hour of Day", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Hour (24-hour clock)", fontweight="bold")
    ax2.set_ylabel("Revenue (£)", fontweight="bold")
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "revenue_by_day_and_hour.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    
    # 3. Top 15 products by revenue and invoice frequency
    top_rev_prod = df_all.groupby(["StockCode", "Description"])["TotalPrice"].sum().reset_index().sort_values("TotalPrice", ascending=False).head(15)
    top_inv_prod = df_all.groupby(["StockCode", "Description"])["InvoiceNo"].nunique().reset_index().sort_values("InvoiceNo", ascending=False).head(15)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.barplot(data=top_rev_prod, y="Description", x="TotalPrice", ax=ax1, palette="rocket")
    ax1.set_title("Top 15 Products by Total Revenue (£)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Revenue (£)", fontweight="bold")
    ax1.set_ylabel("")
    
    sns.barplot(data=top_inv_prod, y="Description", x="InvoiceNo", ax=ax2, palette="mako")
    ax2.set_title("Top 15 Products by Order Frequency (Unique Invoices)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Invoice Count", fontweight="bold")
    ax2.set_ylabel("")
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "top15_products.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    
    # 4. Top 10 countries by revenue and customer counts
    country_stats = df_all.groupby("Country").agg(
        revenue=("TotalPrice", "sum"),
        customers=("CustomerID", "nunique")
    ).sort_values("revenue", ascending=False).head(10).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=country_stats, y="Country", x="revenue", ax=ax1, palette="crest")
    ax1.set_title("Top 10 Countries by Revenue (£)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Revenue (£)", fontweight="bold")
    ax1.set_ylabel("")
    
    country_cust = df_cust.groupby("Country")["CustomerID"].nunique().sort_values(ascending=False).head(10).reset_index()
    sns.barplot(data=country_cust, y="Country", x="CustomerID", ax=ax2, palette="flare")
    ax2.set_title("Top 10 Countries by Unique Customer Base", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Customers", fontweight="bold")
    ax2.set_ylabel("")
    plt.tight_layout()
    fig4_path = FIGURES_DIR / "top10_countries.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    
    # 5. Customer Spend Distribution and Pareto Curve
    cust_spend = df_cust.groupby("CustomerID")["TotalPrice"].sum().sort_values(ascending=False).reset_index()
    cust_spend["cum_revenue"] = cust_spend["TotalPrice"].cumsum()
    cust_spend["cum_revenue_pct"] = (cust_spend["cum_revenue"] / cust_spend["TotalPrice"].sum()) * 100
    cust_spend["cum_customers_pct"] = (np.arange(1, len(cust_spend) + 1) / len(cust_spend)) * 100
    
    top_20_pct_cust_idx = int(np.ceil(0.20 * len(cust_spend)))
    top_20_rev_share = cust_spend.iloc[top_20_pct_cust_idx - 1]["cum_revenue_pct"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Log-scale histogram
    sns.histplot(cust_spend["TotalPrice"], bins=50, log_scale=True, ax=ax1, color="#2980b9", kde=True)
    ax1.set_title("Customer Spend Distribution (Log10 Scale)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Total Spend (£, Log Scale)", fontweight="bold")
    ax1.set_ylabel("Customer Count", fontweight="bold")
    
    # Pareto curve
    ax2.plot(cust_spend["cum_customers_pct"], cust_spend["cum_revenue_pct"], color="#e67e22", linewidth=2.5, label="Cumulative Revenue %")
    ax2.axvline(20, color="#7f8c8d", linestyle="--", alpha=0.7, label=f"Top 20% Customers: {top_20_rev_share:.1f}% Rev")
    ax2.axhline(top_20_rev_share, color="#7f8c8d", linestyle="--", alpha=0.7)
    ax2.plot(20, top_20_rev_share, "ro", markersize=8)
    ax2.set_title("Customer Spend Pareto Curve (80/20 Rule)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Cumulative % of Customers", fontweight="bold")
    ax2.set_ylabel("Cumulative % of Revenue", fontweight="bold")
    ax2.legend(loc="lower right")
    plt.tight_layout()
    fig5_path = FIGURES_DIR / "customer_spend_and_pareto.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    
    # 6. Orders per customer & Basket Size
    orders_per_cust = df_cust.groupby("CustomerID")["InvoiceNo"].nunique()
    repeat_customers = (orders_per_cust >= 2).sum()
    repeat_rate = (repeat_customers / len(orders_per_cust)) * 100
    
    basket_sizes = df_all.groupby("InvoiceNo")["StockCode"].nunique()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(orders_per_cust[orders_per_cust <= 30], bins=30, ax=ax1, color="#8e44ad", discrete=True)
    ax1.set_title(f"Orders per Customer (Repeat Rate: {repeat_rate:.1f}%)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Orders", fontweight="bold")
    ax1.set_ylabel("Customers", fontweight="bold")
    
    sns.histplot(basket_sizes[basket_sizes <= 50], bins=50, ax=ax2, color="#16a085")
    ax2.set_title(f"Basket Size Distribution (Median: {basket_sizes.median():.0f} items)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Unique Products per Invoice", fontweight="bold")
    ax2.set_ylabel("Invoice Count", fontweight="bold")
    plt.tight_layout()
    fig6_path = FIGURES_DIR / "orders_and_basket_size.png"
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    
    # 7. Cancellations / Return Rate
    total_invoices_raw = df_all["InvoiceNo"].nunique() + cancellations_df["InvoiceNo"].nunique()
    return_rate_invoices = (cancellations_df["InvoiceNo"].nunique() / total_invoices_raw) * 100
    return_rate_lines = (len(cancellations_df) / (len(df_all) + len(cancellations_df))) * 100
    
    # Outlier documentation
    max_order = df_all.loc[df_all["Quantity"].idxmax()]
    outlier_info = {
        "stock_code": str(max_order["StockCode"]),
        "description": str(max_order["Description"]),
        "quantity": int(max_order["Quantity"]),
        "unit_price": float(max_order["UnitPrice"]),
        "total_price": float(max_order["TotalPrice"]),
        "invoice_no": str(max_order["InvoiceNo"]),
        "date": str(max_order["InvoiceDate"])
    }
    
    metrics = {
        "total_revenue_all": float(df_all["TotalPrice"].sum()),
        "total_revenue_cust": float(df_cust["TotalPrice"].sum()),
        "total_invoices_all": int(df_all["InvoiceNo"].nunique()),
        "total_customers": int(len(orders_per_cust)),
        "repeat_customers": int(repeat_customers),
        "repeat_rate_pct": round(repeat_rate, 2),
        "median_basket_size": float(basket_sizes.median()),
        "mean_basket_size": round(float(basket_sizes.mean()), 2),
        "max_basket_size": int(basket_sizes.max()),
        "median_customer_spend": round(float(cust_spend["TotalPrice"].median()), 2),
        "mean_customer_spend": round(float(cust_spend["TotalPrice"].mean()), 2),
        "top_20_rev_share_pct": round(top_20_rev_share, 2),
        "return_rate_invoices_pct": round(return_rate_invoices, 2),
        "return_rate_lines_pct": round(return_rate_lines, 2),
        "cancellations_total_rows": int(len(cancellations_df)),
        "cancellations_unique_invoices": int(cancellations_df["InvoiceNo"].nunique()),
        "outlier_wholesale_order": outlier_info
    }
    
    # Build behavior report
    report_path = REPORTS_DIR / "behaviour_report.md"
    content = f"""# Stage 2 Customer Behaviour and Exploratory Data Analysis Report

## 1. Commercial Performance Overview
- **Total Net Revenue Analyzed:** **£{metrics['total_revenue_all']:,.2f}** (Registered Customer Revenue: **£{metrics['total_revenue_cust']:,.2f}**)
- **Total Valid Sales Invoices:** **{metrics['total_invoices_all']:,}**
- **Unique Registered Customers:** **{metrics['total_customers']:,}**
- **Repeat Purchase Rate:** **{metrics['repeat_rate_pct']}%** ({metrics['repeat_customers']:,} customers made 2 or more separate purchases).

---

## 2. Customer Spend Dynamics & The 80/20 Pareto Concentration
- **Median Customer Lifetime Spend:** **£{metrics['median_customer_spend']:,}**
- **Mean Customer Lifetime Spend:** **£{metrics['mean_customer_spend']:,}**
- **Pareto Concentration:** The top **20% of customers generate {metrics['top_20_rev_share_pct']}% of total revenue**, confirming extreme reliance on high-volume wholesale accounts.
- **Spend Distribution:** Highly right-skewed with extreme positive tail, necessitating log-transformation (`log1p`) before distance-based clustering algorithms (K-Means / Hierarchical).

---

## 3. Basket Size and Order Velocity
- **Average Basket Size:** **{metrics['mean_basket_size']} items per invoice** (Median: **{metrics['median_basket_size']:.0f} items**, Maximum single invoice: **{metrics['max_basket_size']:,} distinct items**).
- **Cancellations and Returns:**
  - Cancelled transactions logged: **{metrics['cancellations_total_rows']:,} rows** across **{metrics['cancellations_unique_invoices']:,} invoices**.
  - Return rate by unique invoice: **{metrics['return_rate_invoices_pct']}%**.
  - Return rate by line items: **{metrics['return_rate_lines_pct']}%**.

---

## 4. Extreme Wholesale Outlier Case Documentation
In accordance with Stage 2 specifications, extreme wholesale orders are retained rather than discarded to preserve genuine business reality:
- **Maximum Transaction:** Invoice `{metrics['outlier_wholesale_order']['invoice_no']}` on `{metrics['outlier_wholesale_order']['date']}`.
- **Product:** `{metrics['outlier_wholesale_order']['stock_code']}` — *{metrics['outlier_wholesale_order']['description']}*.
- **Order Volume:** **{metrics['outlier_wholesale_order']['quantity']:,} units** at **£{metrics['outlier_wholesale_order']['unit_price']}** per unit, yielding a single transaction value of **£{metrics['outlier_wholesale_order']['total_price']:,.2f}**.
- **Modeling Mitigation:** Feature engineering applies logarithmic scaling (`log1p(Monetary)`) to prevent single-order distortion during distance metrics calculation.

---

## 5. Visual Artifacts Produced
The following figures have been generated and archived in [`reports/figures/`](file:///d:/MyProjectsForFun/Data%20Mining%20Project/reports/figures/):
1. **`monthly_revenue_and_counts.png`**: Monthly revenue seasonality, order volume, and customer influx showing strong Q4 holiday ramp-up.
2. **`revenue_by_day_and_hour.png`**: Operating week traffic distribution showing peak ordering on Thursday and peak hours between 10:00 and 15:00 UTC (Saturday closed).
3. **`top15_products.png`**: Highest revenue and most frequently purchased giftware catalog items.
4. **`top10_countries.png`**: International market penetration across UK, Netherlands, EIRE, Germany, and France.
5. **`customer_spend_and_pareto.png`**: Log-normal spend distribution alongside the empirical cumulative Pareto curve.
6. **`orders_and_basket_size.png`**: Distributions of purchase frequency and line item diversity per basket.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Behaviour report saved to {report_path}.")
    return metrics
