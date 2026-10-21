import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR
from src.recommender import load_recommender

# Page Configuration
st.set_page_config(
    page_title="RetailMind | Customer Segmentation & Recommender AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2980b9;
        margin-bottom: 12px;
    }
    .badge-champ { background-color: #27ae60; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-loyal { background-color: #2980b9; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-potential { background-color: #f39c12; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-atrisk { background-color: #c0392b; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_recommender_system():
    return load_recommender()

@st.cache_data
def get_segments_data():
    return pd.read_parquet(PROCESSED_DIR / "customer_segments.parquet")

@st.cache_data
def get_global_rules():
    return pd.read_parquet(MODELS_DIR / "rules_global.parquet")

def main():
    st.sidebar.title("🛍️ RetailMind AI")
    st.sidebar.markdown("**E-Commerce Customer Behavior & Recommendation System**")
    
    nav = st.sidebar.radio(
        "Navigation", 
        ["Personalized Recommender", "Customer Segmentation", "Association Rules Finder", "Offline Evaluation Benchmark"]
    )
    
    rec = get_recommender_system()
    df_segments = get_segments_data()
    df_rules = get_global_rules()
    
    # -------------------------------------------------------------
    # TAB 1: PERSONALIZED RECOMMENDER
    # -------------------------------------------------------------
    if nav == "Personalized Recommender":
        st.title("🎯 Personalized Product Recommendation Engine")
        st.markdown("Blends **segment-specific association rules**, **global cross-sell affinities**, and **tiered popularity** to produce explainable suggestions.")
        
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
        
        with col_ctrl1:
            # Provide sample customers from different segments
            example_cids = {
                "Champion (High Spend & Frequency)": 12352,
                "Loyal Customer (Steady Buyer)": 12346,
                "Potential / Promising (Recent)": int(df_segments[df_segments["segment_name"] == "Potential / Promising"]["CustomerID"].iloc[0]),
                "At-Risk / Hibernating": int(df_segments[df_segments["segment_name"] == "At-Risk / Hibernating"]["CustomerID"].iloc[0])
            }
            preset_choice = st.selectbox("Choose a Persona Preset or Enter Custom ID:", list(example_cids.keys()))
            selected_preset_cid = example_cids[preset_choice]
            
            customer_id = st.number_input("Customer ID", value=selected_preset_cid, step=1)
            
        with col_ctrl2:
            top_k = st.slider("Recommendations (K)", min_value=3, max_value=20, value=8)
            
        with col_ctrl3:
            exclude_prev = st.checkbox("Exclude Purchased Items", value=True)
            use_seg_rules = st.checkbox("Enable Segment Rules", value=True)
            
        # Display Customer Profile Card
        st.markdown("---")
        if customer_id in df_segments["CustomerID"].values:
            cust_row = df_segments[df_segments["CustomerID"] == customer_id].iloc[0]
            seg = cust_row["segment_name"]
            
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            col_p1.metric("Customer Segment", seg)
            col_p2.metric("Recency", f"{int(cust_row['Recency'])} days ago")
            col_p3.metric("Purchase Frequency", f"{int(cust_row['Frequency'])} orders")
            col_p4.metric("Total Spend", f"£{cust_row['Monetary']:,.2f}")
            
            # Show past purchases preview
            history = rec.cust_history.get(customer_id, set())
            with st.expander(f"📦 Customer Purchase History ({len(history)} products bought in training period)"):
                past_items = [f"• **{code}** — {rec.desc_map.get(code, 'UNKNOWN')}" for code in list(history)[:25]]
                st.markdown("\n".join(past_items))
                if len(history) > 25:
                    st.caption(f"...and {len(history) - 25} more items.")
        else:
            st.info("ℹ️ New / Cold-Start Customer: Falling back to global retail popularity.")
            
        # Generate Recommendations
        with st.spinner("Generating personalized rankings..."):
            recs_df = rec.recommend(customer_id, K=top_k, exclude_purchased=exclude_prev, use_segment_rules=use_seg_rules)
            
        if not recs_df.empty:
            st.subheader(f"Top {len(recs_df)} Recommendations for Customer {customer_id}")
            
            for _, row in recs_df.iterrows():
                source_color = {
                    "segment_rule": "#27ae60",
                    "global_rule": "#2980b9",
                    "segment_popularity": "#f39c12",
                    "global_popularity": "#7f8c8d"
                }.get(row["source"], "#333333")
                
                with st.container():
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin: 0; color: #2c3e50;">#{row['rank']} {row['description']} <span style="font-size: 13px; color: #7f8c8d;">({row['stock_code']})</span></h4>
                            <span style="background-color: {source_color}; color: white; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;">
                                {row['source'].replace('_', ' ').upper()}
                            </span>
                        </div>
                        <p style="margin: 6px 0 0 0; color: #555; font-size: 14px;"><strong>Reason:</strong> {row['reason']}</p>
                        <p style="margin: 4px 0 0 0; color: #888; font-size: 12px;">Recommendation Score: {row['score']:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No recommendations could be generated with the current constraints.")

    # -------------------------------------------------------------
    # TAB 2: CUSTOMER SEGMENTATION
    # -------------------------------------------------------------
    elif nav == "Customer Segmentation":
        st.title("👥 Customer RFM Segmentation & Behavior Profiling")
        st.markdown("Unsupervised segmentation using **K-Means ($k=4$)** on log-standardized Recency, Frequency, and Monetary attributes.")
        
        # Segment KPI cards
        col1, col2, col3, col4 = st.columns(4)
        for i, (col, seg_name) in enumerate(zip([col1, col2, col3, col4], ["Champions", "Loyal Customers", "Potential / Promising", "At-Risk / Hibernating"])):
            sub = df_segments[df_segments["segment_name"] == seg_name]
            col.metric(
                seg_name,
                f"{len(sub):,} users ({len(sub)/len(df_segments)*100:.1f}%)",
                f"£{sub['Monetary'].mean():,.0f} avg spend"
            )
            
        st.markdown("---")
        
        tab_v1, tab_v2 = st.tabs(["2D PCA Spatial Distribution", "Segment Profile Comparison"])
        
        with tab_v1:
            fig_pca = px.scatter(
                df_segments, x="pca1", y="pca2", color="segment_name",
                hover_data=["CustomerID", "Recency", "Frequency", "Monetary"],
                title="Customer Segments 2D Projection (PCA)",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_pca.update_layout(height=550)
            st.plotly_chart(fig_pca, use_container_width=True)
            
        with tab_v2:
            agg_table = df_segments.groupby("segment_name").agg(
                Customers=("CustomerID", "count"),
                Recency_Days=("Recency", "mean"),
                Order_Frequency=("Frequency", "mean"),
                Avg_Monetary=("Monetary", "mean"),
                Total_Monetary=("Monetary", "sum")
            ).reset_index()
            agg_table["Revenue_Share_%"] = (agg_table["Total_Monetary"] / agg_table["Total_Monetary"].sum()) * 100
            st.dataframe(agg_table.style.format({
                "Customers": "{:,}",
                "Recency_Days": "{:.1f}",
                "Order_Frequency": "{:.1f}",
                "Avg_Monetary": "£{:,.2f}",
                "Total_Monetary": "£{:,.2f}",
                "Revenue_Share_%": "{:.1f}%"
            }), use_container_width=True)
            
            st.markdown("""
            ### Strategic Marketing Playbook:
            - **🏆 Champions (16.7% users, 64.5% revenue):** VIP reward programs, early product releases, dedicated account managers.
            - **💎 Loyal Customers (28.3% users, 22.6% revenue):** Cross-selling campaigns, bulk volume discounts, loyalty incentives.
            - **🌟 Potential / Promising (17.7% users, 7.1% revenue):** Welcome onboarding series, time-sensitive second-order vouchers.
            - **⚠️ At-Risk / Hibernating (37.3% users, 5.9% revenue):** Automated win-back emails, satisfaction surveys, clearance promotions.
            """)

    # -------------------------------------------------------------
    # TAB 3: ASSOCIATION RULES FINDER
    # -------------------------------------------------------------
    elif nav == "Association Rules Finder":
        st.title("🔗 Product Association & Market Basket Mining")
        st.markdown("Mined using **FP-Growth** with minimum confidence $\ge 30\%$ and lift $\ge 1.5$.")
        
        search_query = st.text_input("Filter rules by product keyword (e.g., 'TEA', 'HEART', 'CUP', 'BAG'):", value="TEA")
        
        filtered_rules = df_rules[
            df_rules["antecedent_names"].str.contains(search_query, case=False, na=False) |
            df_rules["consequent_name"].str.contains(search_query, case=False, na=False)
        ]
        
        st.subheader(f"Matching Association Rules ({len(filtered_rules):,} rules found)")
        st.dataframe(filtered_rules[[
            "antecedent_names", "consequent_name", "support", "confidence", "lift", "leverage"
        ]].head(50).style.format({
            "support": "{:.3%}",
            "confidence": "{:.1%}",
            "lift": "{:.2f}",
            "leverage": "{:.4f}"
        }), use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: OFFLINE EVALUATION BENCHMARK
    # -------------------------------------------------------------
    elif nav == "Offline Evaluation Benchmark":
        st.title("📊 Rigorous Offline Evaluation Benchmark")
        st.markdown("""
        All recommendation models evaluated strictly on **unseen test transactions** (`2011-10-01` to `2011-12-09`) 
        under **Protocol B (New Items Only)** across **1,838 overlapping customers**.
        """)
        
        benchmark_data = {
            "Model": [
                "1. Global Popularity",
                "2. Segment Popularity",
                "3. Global Rules Only",
                "4. Ablation (Hybrid w/o Seg Rules)",
                "5. Proposed (Hybrid Complete)"
            ],
            "Hit-Rate@10": [0.2748, 0.3020, 0.3482, 0.3950, 0.4086],
            "Precision@10": [0.0393, 0.0429, 0.0581, 0.0653, 0.0684],
            "Recall@10": [0.0162, 0.0192, 0.0291, 0.0343, 0.0382],
            "NDCG@10": [0.0418, 0.0466, 0.0691, 0.0756, 0.0782],
            "Catalog_Coverage_%": [1.97, 2.34, 6.48, 6.62, 10.99],
            "95% CI (Hit-Rate@10)": [
                "[0.2552, 0.2965]", "[0.2807, 0.3221]", "[0.3281, 0.3700]", "[0.3738, 0.4168]", "[0.3852, 0.4309]"
            ]
        }
        df_bench = pd.DataFrame(benchmark_data)
        
        st.dataframe(df_bench.style.format({
            "Hit-Rate@10": "{:.3f}",
            "Precision@10": "{:.3f}",
            "Recall@10": "{:.3f}",
            "NDCG@10": "{:.3f}",
            "Catalog_Coverage_%": "{:.2f}%"
        }), use_container_width=True)
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            fig_hit = px.bar(
                df_bench, x="Model", y="Hit-Rate@10", color="Model",
                title="Hit-Rate@10 by Model (Higher is better)",
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig_hit.update_layout(showlegend=False, xaxis_tickangle=-30)
            st.plotly_chart(fig_hit, use_container_width=True)
            
        with col_b2:
            fig_cov = px.bar(
                df_bench, x="Model", y="Catalog_Coverage_%", color="Model",
                title="Catalog Coverage (%) — Diversity of Suggestions",
                color_discrete_sequence=px.colors.sequential.Teal_r
            )
            fig_cov.update_layout(showlegend=False, xaxis_tickangle=-30)
            st.plotly_chart(fig_cov, use_container_width=True)

if __name__ == "__main__":
    main()
