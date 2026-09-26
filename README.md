# 🤖 AI Agent & Machine Learning Solutions Suite

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Project_Status-Completed_✅-brightgreen.svg)]()

An enterprise-ready **AI Agent & Machine Learning Solutions Suite** built as a full internship capstone project. The project integrates predictive ML models, autonomous natural language decision agents, automated exploratory data analytics, and an interactive Streamlit UI web application based on a 9-week project milestone roadmap.

---

## 📈 Project Milestones & Internship Roadmap

| Milestone Phase | Duration | Core Deliverables & Key Objectives | Status |
| :--- | :--- | :--- | :---: |
| **Week 1: Project Setup & Planning** | Week 1 | Environment configuration, directory setup, requirements definition, Git initialization. | ✅ Done |
| **Week 2-3: Research & Design** | Weeks 2–3 | Literature research, dataset schema specification, system architecture design. | ✅ Done |
| **Week 4-6: Development Phase 1** | Weeks 4–6 | EDA engine, Sales Forecasting model, Recommendation Agent, K-Means Clustering. | ✅ Done |
| **Week 7-8: Development Phase 2** | Weeks 7–8 | Natural Language Data Query Agent, Autonomous Strategic Advisor, Streamlit UI. | ✅ Done |
| **Week 8-9: Final Review & Submission** | Weeks 8–9 | Comprehensive phase documentation, testing, presentation, GitHub delivery. | ✅ Done |

---

## 🌟 System Features & Capabilities

### 1. 🤖 Autonomous AI Agent Suite
- **Natural Language Data Query Agent (`DataQueryAgent`)**: Answers user queries in plain English (e.g. *"What is total revenue?"*, *"Which product is the top seller?"*, *"Show regional sales"*) and outputs computed analytical results.
- **Autonomous Strategic Advisor (`StrategicDecisionAgent`)**: Evaluates business telemetry, flags low-margin items, analyzes churn risks, and outputs strategic recommendations.

### 2. 🔮 Machine Learning Models
- **Predictive Sales Forecasting (`SalesForecaster`)**: Random Forest Regression model predicting revenue/demand with feature importances and evaluation metrics ($R^2$, MAE, RMSE).
- **Product Recommendation Engine (`RecommendationAgent`)**: Item-Item and User-Item Collaborative Filtering using Cosine Similarity matrices.
- **Customer Behavioral Segmentation (`CustomerSegmenter`)**: K-Means clustering categorizing customers into High-Value VIPs, Steady Growth Buyers, and Occasional Customers.

### 3. 📊 Automated EDA & Visualizations
- Monthly sales & profit growth trend analysis.
- Product category breakdowns with margin percentages.
- Feature correlation heatmaps.

### 4. 💻 Interactive Web UI (Streamlit)
- Sleek dark glassmorphic interface with 5 interactive navigation tabs.
- Live forecasting interactive slider controls.
- Documentation viewer for phase reports.

---

## 🏗️ System Architecture

```
                          +---------------------------------------+
                          |        Streamlit Web Interface        |
                          |               (src/app.py)            |
                          +-------------------+-------------------+
                                              |
      +---------------------------------------+---------------------------------------+
      |                                       |                                       |
      v                                       v                                       v
+-----------------------+           +-----------------------+           +-----------------------+
|  Exploratory EDA      |           |  AI Decision Agents   |           |  Machine Learning     |
|  - Revenue Trends     |           |  - NL Query Engine    |           |  - Sales Forecasting  |
|  - Category Charts    |           |  - Strategic Advisor  |           |  - Collaborative Recs |
|  - Correlation        |           |                       |           |  - K-Means Clustering |
+-----------+-----------+           +-----------+-----------+           +-----------+-----------+
            |                                   |                                   |
            +-----------------------------------+-----------------------------------+
                                                |
                                                v
                                +-------------------------------+
                                |  Data Processing Engine       |
                                |  (src/data_loader.py)         |
                                +---------------+---------------+
                                                |
                                                v
                                +-------------------------------+
                                |  Transactional Dataset        |
                                |  (data/sales_data.csv)        |
                                +-------------------------------+
```

---

## 📂 Project Directory Structure

```
AI AGENTS ML Solutions/
├── data/
│   └── sales_data.csv                  # Transactional sales dataset
├── docs/
│   ├── 01_project_setup_and_planning.md
│   ├── 02_research_and_system_design.md
│   ├── 03_development_phase_1.md
│   ├── 04_development_phase_2.md
│   ├── 05_final_review_and_submission.md
│   └── system_architecture.md
├── reports/
│   └── final_project_report.md         # Final internship report
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Dataset loading & preprocessing
│   ├── eda_engine.py                   # Automated EDA visualizations
│   ├── ml_models.py                    # Forecasting, Recs & Clustering ML
│   ├── ai_agent.py                     # Autonomous AI query & decision agent
│   └── app.py                          # Main Streamlit web application
├── requirements.txt                    # Project dependencies
└── README.md                           # GitHub README documentation
```

---

## 🚀 Installation & Running Locally

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/kartheek2805/AI-Agent-Machine-Learning-Solutions.git
   cd "AI-Agent-Machine-Learning-Solutions"
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit Web Application**:
   ```bash
   streamlit run src/app.py
   ```

---

## 👤 Developer / Intern Information
- **Name**: Kambham Kartheek Reddy
- **GitHub**: [@kartheek2805](https://github.com/kartheek2805)
- **Project**: Internship Project on AI Agent & Machine Learning Solutions