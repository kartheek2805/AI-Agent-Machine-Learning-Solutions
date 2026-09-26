import pandas as pd
import numpy as np

class DataQueryAgent:
    """
    Natural Language Query & Analytical Intelligence Agent.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def answer_query(self, query: str) -> dict:
        q = query.lower().strip()
        
        if self.df is None or self.df.empty:
            return {"answer": "No data is loaded.", "type": "text"}

        if any(w in q for w in ['total revenue', 'total sales', 'how much revenue']):
            val = self.df['Total_Revenue'].sum()
            return {
                "answer": f"💰 The total generated revenue across all transactions is **${val:,.2f}**.",
                "type": "metric",
                "value": f"${val:,.2f}"
            }

        if any(w in q for w in ['total profit', 'net profit', 'how much profit']):
            val = self.df['Profit'].sum()
            return {
                "answer": f"📈 The total profit generated is **${val:,.2f}** (Average Profit Margin: **{(val/self.df['Total_Revenue'].sum())*100:.1f}%**).",
                "type": "metric",
                "value": f"${val:,.2f}"
            }

        if any(w in q for w in ['top product', 'best selling product', 'highest revenue product']):
            top = self.df.groupby('Product_Name')['Total_Revenue'].sum().reset_index().sort_values(by='Total_Revenue', ascending=False).iloc[0]
            return {
                "answer": f"🏆 The top product by revenue is **{top['Product_Name']}** generating **${top['Total_Revenue']:,.2f}**.",
                "type": "metric",
                "value": top['Product_Name']
            }

        if any(w in q for w in ['best category', 'top category', 'highest profit category']):
            top_cat = self.df.groupby('Category')['Profit'].sum().reset_index().sort_values(by='Profit', ascending=False).iloc[0]
            return {
                "answer": f"⭐ The most profitable category is **{top_cat['Category']}** with a net profit of **${top_cat['Profit']:,.2f}**.",
                "type": "metric",
                "value": top_cat['Category']
            }

        if any(w in q for w in ['region', 'regional sales', 'best region']):
            reg_df = self.df.groupby('Region')['Total_Revenue'].sum().reset_index().sort_values(by='Total_Revenue', ascending=False)
            return {
                "answer": "🌍 **Regional Revenue Summary**:",
                "type": "table",
                "data": reg_df
            }

        if any(w in q for w in ['rating', 'satisfaction', 'customer rating']):
            avg_r = self.df['Customer_Rating'].mean()
            return {
                "answer": f"🌟 The average customer rating across all products is **{avg_r:.2f} / 5.00**.",
                "type": "metric",
                "value": f"{avg_r:.2f} ⭐"
            }

        # General summary default response
        return {
            "answer": f"ℹ️ Custom Analysis Result: Dataset contains **{len(self.df)} transactions**, **{self.df['Product_Name'].nunique()} unique products**, and total revenue of **${self.df['Total_Revenue'].sum():,.2f}**.",
            "type": "text"
        }


class StrategicDecisionAgent:
    """
    Autonomous AI Strategy Decision & Optimization Agent.
    """
    def __init__(self, df: pd.DataFrame, ml_metrics: dict = None):
        self.df = df
        self.ml_metrics = ml_metrics or {}

    def generate_strategic_insights() -> list:
        pass

    def generate_strategic_insights(self) -> list:
        insights = []

        if self.df is None or self.df.empty:
            return ["Load dataset to enable autonomous strategic analysis."]

        # 1. Product Margin Bottleneck Analysis
        prod_margins = self.df.groupby('Product_Name').agg({
            'Total_Revenue': 'sum',
            'Profit': 'sum'
        }).reset_index()
        prod_margins['Margin_%'] = (prod_margins['Profit'] / prod_margins['Total_Revenue']) * 100
        lowest_margin = prod_margins.sort_values(by='Margin_%').iloc[0]

        insights.append({
            "topic": "Pricing & Margin Optimization",
            "level": "Warning",
            "icon": "⚠️",
            "finding": f"Product '{lowest_margin['Product_Name']}' has the lowest profit margin at {lowest_margin['Margin_%']:.1f}%.",
            "recommendation": "Negotiate supplier unit costs or increase price by 8-12% to align with portfolio margins."
        })

        # 2. Regional Expansion Opportunity
        reg_df = self.df.groupby('Region')['Total_Revenue'].sum().reset_index().sort_values(by='Total_Revenue')
        lowest_reg = reg_df.iloc[0]
        highest_reg = reg_df.iloc[-1]

        insights.append({
            "topic": "Regional Growth Strategy",
            "level": "Opportunity",
            "icon": "🚀",
            "finding": f"'{highest_reg['Region']}' dominates revenue (${highest_reg['Total_Revenue']:,.0f}), while '{lowest_reg['Region']}' lags behind (${lowest_reg['Total_Revenue']:,.0f}).",
            "recommendation": f"Launch targeted marketing campaigns in {lowest_reg['Region']} utilizing recommendations from high-performing bundles in {highest_reg['Region']}."
        })

        # 3. Machine Learning Forecasting Integrity
        if 'r2' in self.ml_metrics:
            insights.append({
                "topic": "ML Forecasting Integrity",
                "level": "Success",
                "icon": "🤖",
                "finding": f"Sales forecasting model trained with R² score of {self.ml_metrics['r2']*100:.1f}% and RMSE of ${self.ml_metrics.get('rmse', 0)}.",
                "recommendation": "Deploy model predictions to inventory management to automate dynamic stock reordering."
            })

        # 4. Customer Churn & Retention
        cust_ratings = self.df.groupby('Customer_ID')['Customer_Rating'].mean()
        low_rated = len(cust_ratings[cust_ratings < 3.5])
        
        insights.append({
            "topic": "Customer Loyalty & Churn Risk",
            "level": "Action Needed",
            "icon": "👥",
            "finding": f"{low_rated} customers gave ratings below 3.5 stars.",
            "recommendation": "Deploy the Recommendation Agent to send personalized retention discounts for top-rated complement products."
        })

        return insights
