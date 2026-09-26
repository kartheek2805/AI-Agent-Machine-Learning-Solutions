# Week 7-8: Development Phase 2

## Milestones Achieved
During Development Phase 2 (Weeks 7–8), advanced agent components were built, integrated into the system pipeline, and coupled with an interactive web UI.

## Key Features Developed
1. **Natural Language Query Agent (`src/ai_agent.py -> DataQueryAgent`)**:
   - Parses text input queries regarding sales, revenue, products, profit, regions, and ratings.
   - Computes real-time responses and returns formatted text, metrics, or tables.

2. **Strategic Autonomous Decision Agent (`src/ai_agent.py -> StrategicDecisionAgent`)**:
   - Analyzes product profit margin bottlenecks.
   - Evaluates regional revenue disparities.
   - Validates machine learning forecasting model health.
   - Identifies low-satisfaction customer groups and suggests retention tactics.

3. **Streamlit Web Application UI (`src/app.py`)**:
   - Built custom CSS dark glassmorphic interface with 5 interactive navigation tabs.
   - Embedded interactive Plotly/Matplotlib charts and live simulation controls.
