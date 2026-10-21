import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
from src.config import (
    PROCESSED_DIR, MODELS_DIR, W_RULE_SEGMENT, W_RULE_GLOBAL, 
    W_SEG_POPULARITY, W_GLOB_POPULARITY, DEFAULT_K, EXCLUDE_PURCHASED
)

class HybridRecommender:
    def __init__(
        self,
        train_df: pd.DataFrame,
        customer_segments_df: pd.DataFrame,
        global_rules: pd.DataFrame,
        segment_rules_dict: Dict[str, pd.DataFrame],
        w_rule_seg: float = W_RULE_SEGMENT,
        w_rule_glob: float = W_RULE_GLOBAL,
        w_seg_pop: float = W_SEG_POPULARITY,
        w_glob_pop: float = W_GLOB_POPULARITY
    ):
        self.train_df = train_df
        self.customer_segments_df = customer_segments_df.set_index("CustomerID")
        self.global_rules = global_rules
        self.segment_rules_dict = segment_rules_dict
        
        self.w_rule_seg = w_rule_seg
        self.w_rule_glob = w_rule_glob
        self.w_seg_pop = w_seg_pop
        self.w_glob_pop = w_glob_pop
        
        # Build product description lookup
        self.desc_map = (
            train_df.groupby("StockCode")["Description"]
            .agg(lambda x: x.mode().iloc[0] if not x.empty else "UNKNOWN")
            .to_dict()
        )
        
        # Precompute purchase history per customer
        self.cust_history = (
            train_df.groupby("CustomerID")["StockCode"]
            .apply(lambda s: set(s))
            .to_dict()
        )
        
        # Precompute global popularity (fraction of customers who bought item)
        n_total_cust = train_df["CustomerID"].nunique()
        glob_counts = train_df.groupby("StockCode")["CustomerID"].nunique()
        self.global_pop_share = (glob_counts / n_total_cust).to_dict()
        
        # Min-max scale global popularity
        max_gp = max(self.global_pop_share.values()) if self.global_pop_share else 1.0
        self.global_pop_norm = {k: v / max_gp for k, v in self.global_pop_share.items()}
        self.global_top_items = glob_counts.sort_values(ascending=False).index.tolist()
        
        # Precompute segment popularity
        self.segment_pop_share: Dict[str, Dict[str, float]] = {}
        self.segment_pop_norm: Dict[str, Dict[str, float]] = {}
        self.segment_top_items: Dict[str, List[str]] = {}
        
        merged_train = train_df.merge(
            customer_segments_df[["CustomerID", "segment_name"]], on="CustomerID", how="inner"
        )
        for seg_name, group in merged_train.groupby("segment_name"):
            n_seg_cust = group["CustomerID"].nunique()
            seg_counts = group.groupby("StockCode")["CustomerID"].nunique()
            share_dict = (seg_counts / n_seg_cust).to_dict()
            self.segment_pop_share[seg_name] = share_dict
            
            max_sp = max(share_dict.values()) if share_dict else 1.0
            self.segment_pop_norm[seg_name] = {k: v / max_sp for k, v in share_dict.items()}
            self.segment_top_items[seg_name] = seg_counts.sort_values(ascending=False).index.tolist()
            
        # Parse and index rules for fast evaluation
        self.parsed_global_rules = self._index_rules(global_rules, "global")
        self.parsed_segment_rules = {
            seg: self._index_rules(rules, "segment") for seg, rules in segment_rules_dict.items()
        }
        
    def _index_rules(self, rules_df: pd.DataFrame, source_type: str) -> List[Dict[str, Any]]:
        """Parses rules into fast-lookup list of dictionaries with antecedent sets."""
        parsed = []
        if rules_df is None or rules_df.empty:
            return parsed
            
        for _, row in rules_df.iterrows():
            ant_set = set(row["antecedents_str"].split(","))
            con = row["consequents_str"]
            conf = float(row["confidence"])
            lift = float(row["lift"])
            ant_names = row.get("antecedent_names", "")
            con_name = row.get("consequent_name", "")
            
            parsed.append({
                "antecedents": ant_set,
                "consequent": con,
                "confidence": conf,
                "lift": lift,
                "rule_strength": conf * lift,
                "source": source_type,
                "ant_names": ant_names,
                "con_name": con_name
            })
        return parsed

    def recommend(
        self, 
        customer_id: int, 
        K: int = DEFAULT_K, 
        exclude_purchased: bool = EXCLUDE_PURCHASED,
        use_segment_rules: bool = True
    ) -> pd.DataFrame:
        """Generates top-K recommendations with cold-start fallbacks and transparent explanations."""
        history = self.cust_history.get(customer_id, set())
        
        # Check segment
        seg_name = None
        if customer_id in self.customer_segments_df.index:
            seg_name = self.customer_segments_df.loc[customer_id, "segment_name"]
            if isinstance(seg_name, pd.Series):
                seg_name = seg_name.iloc[0]
                
        # 1. Rule Matching
        candidate_scores: Dict[str, Dict[str, Any]] = {}
        
        # A) Segment rules
        if use_segment_rules and seg_name and seg_name in self.parsed_segment_rules:
            for rule in self.parsed_segment_rules[seg_name]:
                if rule["antecedents"].issubset(history):
                    item = rule["consequent"]
                    if exclude_purchased and item in history:
                        continue
                    r_score = self.w_rule_seg * rule["rule_strength"]
                    if item not in candidate_scores or r_score > candidate_scores[item]["rule_score"]:
                        candidate_scores[item] = {
                            "rule_score": r_score,
                            "source": "segment_rule",
                            "rule_meta": rule
                        }
                        
        # B) Global rules
        for rule in self.parsed_global_rules:
            if rule["antecedents"].issubset(history):
                item = rule["consequent"]
                if exclude_purchased and item in history:
                    continue
                r_score = self.w_rule_glob * rule["rule_strength"]
                if item not in candidate_scores or r_score > candidate_scores[item]["rule_score"]:
                    candidate_scores[item] = {
                        "rule_score": r_score,
                        "source": "global_rule",
                        "rule_meta": rule
                    }
                    
        # 2. Score Candidates
        results = []
        for item, info in candidate_scores.items():
            r_score = info["rule_score"]
            seg_pop = self.segment_pop_norm.get(seg_name, {}).get(item, 0.0) if seg_name else 0.0
            glob_pop = self.global_pop_norm.get(item, 0.0)
            
            final_score = r_score + (self.w_seg_pop * seg_pop) + (self.w_glob_pop * glob_pop)
            rule = info["rule_meta"]
            reason = (
                f"Because you bought {rule['ant_names']}; customers who bought these also bought this "
                f"{rule['confidence']:.0%} of the time (lift {rule['lift']:.1f})."
            )
            results.append({
                "stock_code": item,
                "description": self.desc_map.get(item, "UNKNOWN"),
                "score": final_score,
                "source": info["source"],
                "reason": reason
            })
            
        results.sort(key=lambda x: x["score"], reverse=True)
        recommended_items = {r["stock_code"] for r in results}
        
        # 3. Cold-Start Fallback 1: Segment Popularity
        if len(results) < K and seg_name and seg_name in self.segment_top_items:
            for item in self.segment_top_items[seg_name]:
                if exclude_purchased and item in history:
                    continue
                if item not in recommended_items:
                    pct = self.segment_pop_share[seg_name].get(item, 0.0) * 100
                    results.append({
                        "stock_code": item,
                        "description": self.desc_map.get(item, "UNKNOWN"),
                        "score": self.w_seg_pop * self.segment_pop_norm[seg_name].get(item, 0.0),
                        "source": "segment_popularity",
                        "reason": f"Popular among {seg_name} customers ({pct:.1f}% of segment bought it)."
                    })
                    recommended_items.add(item)
                    if len(results) >= K:
                        break
                        
        # 4. Cold-Start Fallback 2: Global Popularity
        if len(results) < K:
            for item in self.global_top_items:
                if exclude_purchased and item in history:
                    continue
                if item not in recommended_items:
                    pct = self.global_pop_share.get(item, 0.0) * 100
                    results.append({
                        "stock_code": item,
                        "description": self.desc_map.get(item, "UNKNOWN"),
                        "score": self.w_glob_pop * self.global_pop_norm.get(item, 0.0),
                        "source": "global_popularity",
                        "reason": f"Among the best-selling products overall ({pct:.1f}% of all customers bought it)."
                    })
                    recommended_items.add(item)
                    if len(results) >= K:
                        break
                        
        df_rec = pd.DataFrame(results[:K])
        if not df_rec.empty:
            df_rec["rank"] = np.arange(1, len(df_rec) + 1)
            df_rec = df_rec[["rank", "stock_code", "description", "score", "source", "reason"]]
        return df_rec

def load_recommender() -> HybridRecommender:
    """Factory function to instantiate HybridRecommender from cached parquet models."""
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    customer_segments_df = pd.read_parquet(PROCESSED_DIR / "customer_segments.parquet")
    global_rules = pd.read_parquet(MODELS_DIR / "rules_global.parquet")
    
    # Load segment rules
    segment_rules_dict = {}
    rule_files = list(MODELS_DIR.glob("rules_segment_*.parquet"))
    for rf in rule_files:
        seg_key = rf.stem.replace("rules_segment_", "")
        # Map filename back to segment name
        for real_name in customer_segments_df["segment_name"].unique():
            if real_name.lower().replace(" ", "_").replace("/", "_") == seg_key:
                segment_rules_dict[real_name] = pd.read_parquet(rf)
                break
                
    return HybridRecommender(train_df, customer_segments_df, global_rules, segment_rules_dict)
