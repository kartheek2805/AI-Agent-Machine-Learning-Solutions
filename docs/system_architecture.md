# System Architecture Specification

```
                          +---------------------------------------+
                          |        User / Executive Interface     |
                          |        (Streamlit Web Dashboard)      |
                          +-------------------+-------------------+
                                              |
                                              v
      +---------------------------------------+---------------------------------------+
      |                                       |                                       |
      v                                       v                                       v
+-----------------------+           +-----------------------+           +-----------------------+
|  Data Explorer & EDA  |           |  AI Decision Agents   |           |  Machine Learning     |
|  - Revenue Trends     |           |  - NL Query Engine    |           |  - Sales Forecasting  |
|  - Category Charts    |           |  - Strategic Advisor  |           |  - Collaborative Recs |
|  - Correlations       |           |                       |           |  - K-Means Clustering |
+-----------+-----------+           +-----------+-----------+           +-----------+-----------+
            |                                   |                                   |
            +-----------------------------------+-----------------------------------+
                                                |
                                                v
                                +-------------------------------+
                                |  Data Processing Engine       |
                                |  (Pandas, DataLoader, Clean)  |
                                +---------------+---------------+
                                                |
                                                v
                                +-------------------------------+
                                |  Transactional Dataset        |
                                |  (data/sales_data.csv)        |
                                +-------------------------------+
```

## Layer Description

1. **Presentation Layer**: Streamlit dashboard with custom CSS, dark-mode visuals, interactive sliders, and multi-tab navigation.
2. **AI Agent Layer**: `DataQueryAgent` parses natural language text input, and `StrategicDecisionAgent` generates automated business telemetry insights.
3. **Machine Learning Layer**:
   - `SalesForecaster`: Random Forest Regression for demand & revenue forecasting.
   - `RecommendationAgent`: Cosine similarity collaborative filtering.
   - `CustomerSegmenter`: K-Means clustering for RFM segmentation.
4. **Data Access Layer**: `DataLoader` cleanses, transforms, and validates sales datasets.
