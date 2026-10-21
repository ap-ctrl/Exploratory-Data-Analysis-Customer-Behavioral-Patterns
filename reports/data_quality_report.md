# Stage 1 Data Quality and Understanding Report

## 1. Dataset Identification and Semantic Scope
- **Source:** UCI Machine Learning Repository — *Online Retail* (Daqing Chen et al.)
- **Semantic Definition:** **One row represents one line item of an invoice.** An individual shopping basket (order) corresponds to a unique `InvoiceNo` and may contain multiple line items.
- **Temporal Coverage:** `2010-12-01 08:26:00` to `2011-12-09 12:50:00` (~1 year, 1 week).
- **Core Dimensions:**
  - Total records (line items): **541,909**
  - Features: **8**
  - Unique Invoices: **25,900**
  - Unique Products (StockCodes): **4,070**
  - Unique Registered Customers: **4,372**
  - Unique Countries: **38**

---

## 2. Missing Value Analysis

| Feature | Data Type | Missing Count | Missing Percentage |
| :--- | :--- | :--- | :--- |
| `InvoiceNo` | `string` | 0 | 0.0% |
| `StockCode` | `string` | 0 | 0.0% |
| `Description` | `string` | 1,454 | 0.27% |
| `Quantity` | `int64` | 0 | 0.0% |
| `InvoiceDate` | `datetime64[us]` | 0 | 0.0% |
| `UnitPrice` | `float64` | 0 | 0.0% |
| `CustomerID` | `float64` | 135,080 | 24.93% |
| `Country` | `string` | 0 | 0.0% |

### Observations on Missing Values:
1. **CustomerID (24.93% missing):** Exactly 135,080 rows lack a customer identifier. These correspond to guest checkouts, telephone orders, or unauthenticated online transactions. They are preserved for transaction-level analysis (global association rules and storewide popularity) but excluded for customer-level segmentation (RFM) and personalized evaluation.
2. **Description (0.27% missing):** Exactly 1,454 rows lack product descriptions, often associated with inventory reconciliation or damaged stock.

---

## 3. Data Integrity, Cancellations, and Anomalies

| Category | Count | Percentage of Total Rows |
| :--- | :--- | :--- |
| **Exact Duplicate Rows** | 5,268 | 0.97% |
| **Cancellations (`InvoiceNo` starts with 'C')** | 9,288 | 1.71% |
| **Non-positive Quantities (`Quantity <= 0`)** | 10,624 | 1.96% |
| **Non-positive Unit Prices (`UnitPrice <= 0`)** | 2,517 | 0.46% |

### Non-Product StockCodes Found:
The dataset contains non-merchandise administrative codes that do not represent physical gift items:
| StockCode | Occurrence Count | Description / Role |
| :--- | :--- | :--- |
| `POST` | 1,256 |
| `DOT` | 710 |
| `M` | 571 |
| `C2` | 144 |
| `D` | 77 |
| `S` | 63 |
| `BANK CHARGES` | 37 |
| `AMAZONFEE` | 34 |
| `CRUK` | 16 |
| `DCGSSGIRL` | 13 |
*Additional codes include `AMAZONFEE`, `BANK CHARGES`, `CRUK` (Cancer Research UK), and `S` (samples).*

---

## 4. Dataset Scope and Real-World Limitations
1. **Single Retailer Bias:** Data reflects a single UK-based giftware retailer with heavy seasonal wholesale spikes (notably Q4 Christmas demand).
2. **Wholesale Concentration:** Many transactions involve bulk quantities (up to tens of thousands of units per order) alongside single-item retail purchases, leading to heavy-tailed spend distributions.
3. **No Customer Demographics:** No demographic data (age, gender, income) or customer locations (beyond national IP/country) exist. Customer modeling must rely strictly on behavioral click/purchase telemetry (RFM).
4. **Guest Checkouts:** Approximately one quarter of transactions cannot be attributed to recurring user accounts.
