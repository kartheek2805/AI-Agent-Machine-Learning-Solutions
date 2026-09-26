import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import DataLoader
from src.eda_engine import EDAEngine
from src.ml_models import SalesForecaster, RecommendationAgent, CustomerSegmenter
from src.ai_agent import DataQueryAgent, StrategicDecisionAgent

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="AI Agent & ML Solutions Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark glassmorphic design & milestone rendering
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }
    .stMetric {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .milestone-card {
        background: #1A2332;
        border-left: 6px solid #2ECC71;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .milestone-card h4 {
        color: #2ECC71;
        margin-bottom: 6px;
        font-weight: 600;
    }
    .milestone-card p {
        color: #CBD5E1;
        margin-bottom: 0px;
        font-size: 0.95rem;
    }
    .insight-box {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING & SESSION STATE
# ==========================================
@st.cache_data
def load_project_data():
    loader = DataLoader()
    return loader.load_data(), loader.get_summary_stats()

df, stats = load_project_data()
eda = EDAEngine(df)

# Initialize Models
if 'forecaster' not in st.session_state:
    forecaster = SalesForecaster()
    metrics, feat_imp, X_test, y_test, preds = forecaster.train(df)
    st.session_state['forecaster'] = forecaster
    st.session_state['forecast_metrics'] = metrics
    st.session_state['feat_imp'] = feat_imp
    st.session_state['y_test'] = y_test
    st.session_state['preds'] = preds

if 'recommender' not in st.session_state:
    recommender = RecommendationAgent()
    recommender.fit(df)
    st.session_state['recommender'] = recommender

if 'segmenter' not in st.session_state:
    segmenter = CustomerSegmenter()
    segmented_df = segmenter.segment(df)
    st.session_state['segmented_df'] = segmented_df

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=70)
st.sidebar.title("AI Agent & ML Suite")
st.sidebar.caption("Internship Project & Decision System")

nav_option = st.sidebar.radio(
    "Navigate Project Phases:",
    [
        "📈 Project Milestones",
        "📊 Data Explorer & EDA",
        "🤖 AI Agent & QA Assistant",
        "🔮 Machine Learning Suite",
        "📄 Final Review & Documentation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Status:** `Completed ✅`")
st.sidebar.markdown("**Repository:** `GitHub Origin Main`")

# ==========================================
# TAB 1: PROJECT MILESTONES
# ==========================================
if nav_option == "📈 Project Milestones":
    st.title("📈 Project Milestones & Roadmap")
    st.markdown("Track the 5 execution phases of the **AI Agent & Machine Learning Solutions** internship project.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="milestone-card">
            <h4>✅ Week 1: Project Setup & Planning</h4>
            <p>Understand requirements, set up development environment, create project roadmap, initialize Git repository.</p>
        </div>
        
        <div class="milestone-card">
            <h4>✅ Week 2-3: Research & Design</h4>
            <p>Conduct research, create system design, define technical architecture, format dataset schemas.</p>
        </div>
        
        <div class="milestone-card">
            <h4>✅ Week 4-6: Development Phase 1</h4>
            <p>Implement core features: Sales EDA engine, Predictive Forecasting model, Item Collaborative Recommendation Agent, K-Means Customer Segmenter.</p>
        </div>
        
        <div class="milestone-card">
            <h4>✅ Week 7-8: Development Phase 2</h4>
            <p>Complete advanced features: Autonomous Natural Language Query Agent, Strategic Decision Advisor, interactive Streamlit dashboard.</p>
        </div>
        
        <div class="milestone-card">
            <h4>✅ Week 8-9: Final Review & Submission</h4>
            <p>Documentation, final testing, project presentation, GitHub repository synchronization, final delivery.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.subheader("Progress Summary")
        st.progress(1.0, text="Overall Completion: 100%")
        st.metric("Total Transactions Analyzed", f"{stats.get('total_transactions', 0):,}")
        st.metric("Total Revenue Processed", f"${stats.get('total_revenue', 0):,.2f}")
        st.metric("ML Forecasting R² Score", f"{st.session_state['forecast_metrics'].get('r2', 0)*100:.1f}%")

# ==========================================
# TAB 2: DATA EXPLORER & EDA
# ==========================================
elif nav_option == "📊 Data Explorer & EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Comprehensive view of sales, products, margins, and customer behavior.")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Revenue", f"${stats.get('total_revenue', 0):,.2f}")
    m2.metric("Total Net Profit", f"${stats.get('total_profit', 0):,.2f}")
    m3.metric("Units Sold", f"{stats.get('total_units_sold', 0):,}")
    m4.metric("Avg Rating", f"{stats.get('avg_customer_rating', 0):.2f} ⭐")
    
    st.markdown("---")
    
    t1, t2, t3 = st.tabs(["Monthly Revenue Trend", "Category Breakdown", "Correlation Matrix"])
    
    with t1:
        monthly_df = eda.get_monthly_revenue_trend()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(monthly_df['Month_Name'], monthly_df['Total_Revenue'], marker='o', color='#3B82F6', label='Revenue', linewidth=2.5)
        ax.plot(monthly_df['Month_Name'], monthly_df['Profit'], marker='s', color='#10B981', label='Profit', linewidth=2.5)
        ax.set_title("Monthly Revenue & Profit Growth", fontsize=12, fontweight='bold')
        ax.set_ylabel("USD ($)")
        plt.xticks(rotation=45)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        
    with t2:
        cat_df = eda.get_category_breakdown()
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.dataframe(cat_df, use_container_width=True)
        with col_c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(cat_df['Category'], cat_df['Total_Revenue'], color=['#6366F1', '#8B5CF6', '#EC4899', '#14B8A6'])
            ax.set_title("Revenue by Product Category")
            plt.xticks(rotation=15)
            plt.tight_layout()
            st.pyplot(fig)
            
    with t3:
        corr = eda.get_correlation_matrix()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        plt.tight_layout()
        st.pyplot(fig)

# ==========================================
# TAB 3: AI AGENT & QA ASSISTANT
# ==========================================
elif nav_option == "🤖 AI Agent & QA Assistant":
    st.title("🤖 Autonomous AI Agent & Natural Language Query Assistant")
    st.markdown("Ask questions in plain English or view autonomous AI strategic recommendations.")
    
    col_q1, col_q2 = st.columns([1, 1])
    
    with col_q1:
        st.subheader("💬 Natural Language Query Engine")
        user_query = st.text_input(
            "Ask a question about sales data:",
            value="What is total revenue?",
            placeholder="e.g. Which product is top seller? Or What is average customer rating?"
        )
        
        if st.button("Submit Query to AI Agent"):
            query_agent = DataQueryAgent(df)
            response = query_agent.answer_query(user_query)
            
            st.markdown(f"**AI Agent Response:**\n\n{response['answer']}")
            if response['type'] == 'table':
                st.dataframe(response['data'], use_container_width=True)
                
    with col_q2:
        st.subheader("🧠 Autonomous Strategic Advisor")
        st.caption("AI Agent evaluates business telemetry and flags optimization actions.")
        
        strat_agent = StrategicDecisionAgent(df, st.session_state['forecast_metrics'])
        insights = strat_agent.generate_strategic_insights()
        
        for ins in insights:
            st.markdown(f"""
            <div class="insight-box">
                <h5>{ins['icon']} <b>{ins['topic']}</b> ({ins['level']})</h5>
                <p><b>Finding:</b> {ins['finding']}</p>
                <p><b>Recommendation:</b> {ins['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# TAB 4: MACHINE LEARNING SUITE
# ==========================================
elif nav_option == "🔮 Machine Learning Suite":
    st.title("🔮 Machine Learning Models & Solutions")
    
    ml_tab1, ml_tab2, ml_tab3 = st.tabs([
        "📈 Sales & Demand Forecasting",
        "🎯 Collaborative Recommendation Agent",
        "👥 Customer Segmentation (K-Means)"
    ])
    
    with ml_tab1:
        st.subheader("Random Forest Revenue Forecasting Model")
        metrics = st.session_state['forecast_metrics']
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Model R² Score", f"{metrics['r2']*100:.1f}%")
        k2.metric("Mean Absolute Error (MAE)", f"${metrics['mae']:,.2f}")
        k3.metric("Root Mean Squared Error (RMSE)", f"${metrics['rmse']:,.2f}")
        
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            st.markdown("#### Feature Importance")
            st.dataframe(st.session_state['feat_imp'], use_container_width=True)
        with col_f2:
            st.markdown("#### Test Target vs Predictions")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(st.session_state['y_test'], st.session_state['preds'], alpha=0.6, color='#2563EB')
            ax.plot([st.session_state['y_test'].min(), st.session_state['y_test'].max()],
                    [st.session_state['y_test'].min(), st.session_state['y_test'].max()], 'r--', lw=2)
            ax.set_xlabel("Actual Revenue ($)")
            ax.set_ylabel("Predicted Revenue ($)")
            ax.set_title("Actual vs Predicted Revenue")
            plt.tight_layout()
            st.pyplot(fig)
            
        st.markdown("---")
        st.markdown("#### 🎛️ Interactive Prediction Simulator")
        sim_price = st.slider("Unit Price ($)", 100, 1500, 750)
        sim_units = st.slider("Units Sold", 1, 20, 5)
        
        if st.button("Predict Revenue"):
            forecaster = st.session_state['forecaster']
            predicted_rev = forecaster.predict_future_sale(sim_price, sim_units)
            st.success(f"Estimated Forecasted Revenue: **${predicted_rev:,.2f}**")
            
    with ml_tab2:
        st.subheader("Item-Item Collaborative Filtering Recommendation Agent")
        recommender = st.session_state['recommender']
        
        customers = df['Customer_ID'].unique()
        selected_cust = st.selectbox("Select Customer ID:", customers[:10])
        
        top_recs = recommender.recommend_products_for_user(selected_cust, top_n=3)
        
        st.markdown(f"#### Top Recommended Products for `{selected_cust}`:")
        for rank, (prod, score) in enumerate(top_recs, 1):
            st.markdown(f"**{rank}. {prod}** — Predicted Interest Rating: `{score} / 5.0` ⭐")
            
        st.markdown("---")
        st.markdown("#### User-Item Cosine Similarity Heatmap")
        if recommender.item_similarity is not None:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(recommender.item_similarity, annot=True, cmap="Blues", fmt=".2f", ax=ax)
            ax.set_title("Product Cosine Similarity Matrix")
            plt.tight_layout()
            st.pyplot(fig)
            
    with ml_tab3:
        st.subheader("Customer Behavioral Clustering (K-Means)")
        seg_df = st.session_state['segmented_df']
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(
            data=seg_df,
            x='Total_Revenue',
            y='Order_Frequency',
            hue='Segment_Name',
            palette='Set1',
            s=100,
            ax=ax
        )
        ax.set_title("Customer Segments by Total Revenue & Order Frequency")
        plt.tight_layout()
        st.pyplot(fig)
        
        st.dataframe(seg_df[['Customer_ID', 'Segment_Name', 'Total_Revenue', 'Order_Frequency', 'Customer_Rating']], use_container_width=True)

# ==========================================
# TAB 5: FINAL REVIEW & DOCUMENTATION
# ==========================================
elif nav_option == "📄 Final Review & Documentation":
    st.title("📄 Internship Final Review & Documentation")
    st.markdown("Access project reports, architecture diagrams, and submission deliverables.")
    
    doc_choice = st.selectbox(
        "Select Document to View:",
        [
            "01 Project Setup & Planning (Week 1)",
            "02 Research & System Design (Week 2-3)",
            "03 Development Phase 1 (Week 4-6)",
            "04 Development Phase 2 (Week 7-8)",
            "05 Final Review & Submission (Week 8-9)",
            "System Architecture Specification"
        ]
    )
    
    file_map = {
        "01 Project Setup & Planning (Week 1)": "01_project_setup_and_planning.md",
        "02 Research & System Design (Week 2-3)": "02_research_and_system_design.md",
        "03 Development Phase 1 (Week 4-6)": "03_development_phase_1.md",
        "04 Development Phase 2 (Week 7-8)": "04_development_phase_2.md",
        "05 Final Review & Submission (Week 8-9)": "05_final_review_and_submission.md",
        "System Architecture Specification": "system_architecture.md"
    }
    
    doc_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', file_map[doc_choice])
    if os.path.exists(doc_path):
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        st.markdown(content)
    else:
        st.info("Document file is ready for generation.")
