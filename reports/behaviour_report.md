# Stage 2 Customer Behaviour and Exploratory Data Analysis Report

## 1. Commercial Performance Overview
- **Total Net Revenue Analyzed:** **£10,246,820.87** (Registered Customer Revenue: **£8,737,227.64**)
- **Total Valid Sales Invoices:** **19,773**
- **Unique Registered Customers:** **4,334**
- **Repeat Purchase Rate:** **65.27%** (2,829 customers made 2 or more separate purchases).

---

## 2. Customer Spend Dynamics & The 80/20 Pareto Concentration
- **Median Customer Lifetime Spend:** **£662.57**
- **Mean Customer Lifetime Spend:** **£2,015.97**
- **Pareto Concentration:** The top **20% of customers generate 74.6% of total revenue**, confirming extreme reliance on high-volume wholesale accounts.
- **Spend Distribution:** Highly right-skewed with extreme positive tail, necessitating log-transformation (`log1p`) before distance-based clustering algorithms (K-Means / Hierarchical).

---

## 3. Basket Size and Order Velocity
- **Average Basket Size:** **26.16 items per invoice** (Median: **15 items**, Maximum single invoice: **1,108 distinct items**).
- **Cancellations and Returns:**
  - Cancelled transactions logged: **9,251 rows** across **3,836 invoices**.
  - Return rate by unique invoice: **16.25%**.
  - Return rate by line items: **1.74%**.

---

## 4. Extreme Wholesale Outlier Case Documentation
In accordance with Stage 2 specifications, extreme wholesale orders are retained rather than discarded to preserve genuine business reality:
- **Maximum Transaction:** Invoice `581483` on `2011-12-09 09:15:00`.
- **Product:** `23843` — *PAPER CRAFT , LITTLE BIRDIE*.
- **Order Volume:** **80,995 units** at **£2.08** per unit, yielding a single transaction value of **£168,469.60**.
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
