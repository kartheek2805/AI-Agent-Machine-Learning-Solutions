# Week 2-3: Research & System Design

## Overview
During Weeks 2–3, technical research was conducted on AI agent architectures, collaborative filtering recommendation algorithms, dynamic forecasting models, and UI/UX design patterns.

## Technical Architecture & Pipeline Design
1. **Data Ingestion & Preprocessing Layer**:
   - Parses transactional CSV data containing `Transaction_ID`, `Date`, `Customer_ID`, `Product_ID`, `Unit_Price`, `Units_Sold`, `Total_Revenue`, `Profit`, `Customer_Rating`, `Region`.
   - Handles missing values, enforces numerical types, calculates profit margins, and extracts datetime features.

2. **Machine Learning Engines Layer**:
   - **Sales Forecasting**: Uses Random Forest Regression and Ridge Regression trained on temporal and categorical features. Evaluated using $R^2$, MAE, and RMSE.
   - **Recommendation Agent**: Implements Item-based and User-based Collaborative Filtering utilizing Cosine Similarity matrices over pivot table customer ratings.
   - **Customer Segmentation**: Employs K-Means Clustering on normalized Recency, Frequency, and Monetary (RFM) vectors.

3. **Autonomous AI Agent Layer**:
   - **Natural Language Data Query Agent**: Interprets user questions in English, selects pandas aggregations, and computes answers.
   - **Strategic Decision Advisor**: Evaluates telemetry and generates operational business recommendations.

4. **Interactive Dashboard Layer**:
   - Built on Streamlit with dark-mode aesthetic styling, live Plotly/Matplotlib visualizers, and interactive sliders.
