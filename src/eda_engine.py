import pandas as pd
import numpy as np

class EDAEngine:
    """
    Automated Exploratory Data Analysis & Visualization Module.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def get_monthly_revenue_trend(self) -> pd.DataFrame:
        """
        Aggregates revenue and profit by month.
        """
        if 'Date' not in self.df.columns:
            return pd.DataFrame()
        
        monthly = self.df.set_index('Date').resample('ME').agg({
            'Total_Revenue': 'sum',
            'Profit': 'sum',
            'Units_Sold': 'sum',
            'Transaction_ID': 'count'
        }).reset_index()
        monthly['Month_Name'] = monthly['Date'].dt.strftime('%b %Y')
        return monthly

    def get_category_breakdown(self) -> pd.DataFrame:
        """
        Aggregates metrics by Category.
        """
        if 'Category' not in self.df.columns:
            return pd.DataFrame()
        
        cat_df = self.df.groupby('Category').agg({
            'Total_Revenue': 'sum',
            'Profit': 'sum',
            'Units_Sold': 'sum',
            'Customer_Rating': 'mean',
            'Transaction_ID': 'count'
        }).reset_index().rename(columns={'Transaction_ID': 'Orders'})
        cat_df['Profit_Margin_%'] = (cat_df['Profit'] / cat_df['Total_Revenue']) * 100
        return cat_df.sort_values(by='Total_Revenue', ascending=False)

    def get_top_products(self, n: int = 5) -> pd.DataFrame:
        """
        Retrieves top N products by total revenue.
        """
        if 'Product_Name' not in self.df.columns:
            return pd.DataFrame()
        
        prod_df = self.df.groupby(['Product_ID', 'Product_Name', 'Category']).agg({
            'Total_Revenue': 'sum',
            'Profit': 'sum',
            'Units_Sold': 'sum',
            'Customer_Rating': 'mean'
        }).reset_index()
        return prod_df.sort_values(by='Total_Revenue', ascending=False).head(n)

    def get_regional_distribution(self) -> pd.DataFrame:
        """
        Aggregates performance metrics by Region.
        """
        if 'Region' not in self.df.columns:
            return pd.DataFrame()
        
        reg_df = self.df.groupby('Region').agg({
            'Total_Revenue': 'sum',
            'Profit': 'sum',
            'Units_Sold': 'sum',
            'Transaction_ID': 'count'
        }).reset_index().rename(columns={'Transaction_ID': 'Transactions'})
        return reg_df.sort_values(by='Total_Revenue', ascending=False)

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Returns numeric correlation matrix.
        """
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        return self.df[num_cols].corr().round(3)
