from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_EXCEL_PATH = BASE_DIR / "dataset" / "Online Retail.xlsx"
RAW_PARQUET_PATH = DATA_DIR / "raw_parquet" / "online_retail.parquet"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Reproducibility
RANDOM_STATE = 42

# Time split
SPLIT_DATE = "2011-10-01"

# Cleaning constants
STOCK_CODE_REGEX = r"^\d{5}[A-Za-z]*$"

# Segmentation parameters
K_MIN = 2
K_MAX = 10
KMEANS_N_INIT = 20
RECOMMENDED_K = 4  # Verified via silhouette & elbow

# Association parameters
TOP_N_PRODUCTS = 1000
MIN_SUPPORT_GLOBAL = 0.01
MIN_SUPPORT_FALLBACK = 0.005
MAX_LEN_ITEMSETS = 3
MIN_LIFT = 1.5
MIN_CONFIDENCE = 0.3

# Segment association thresholds (support adapted to smaller segment slices)
SEGMENT_MIN_SUPPORT = {
    "Champions": 0.015,
    "Loyal Customers": 0.01,
    "Potential / Promising": 0.01,
    "At-Risk / Hibernating": 0.01,
}

# Recommender parameters
W_RULE_SEGMENT = 1.0
W_RULE_GLOBAL = 0.7
W_SEG_POPULARITY = 0.5
W_GLOB_POPULARITY = 0.2
DEFAULT_K = 10
EXCLUDE_PURCHASED = True

# Evaluation parameters
EVAL_K_LIST = [5, 10, 20]
BOOTSTRAP_ROUNDS = 1000
