import os
import pandas as pd
import numpy as np

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sales_data.csv')

class DataLoader:
    """
    Handles dataset loading, validation, and preprocessing for the AI & ML Solutions suite.
    """
    def __init__(self, file_path: str = None):
        self.file_path = file_path or DEFAULT_DATA_PATH
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Loads CSV dataset and applies standard preprocessing.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")
        
        self.df = pd.read_csv(self.file_path)
        return self.preprocess()

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocesses dates, missing values, and calculated columns.
        """
        if self.df is None:
            return pd.DataFrame()
        
        # Ensure date format
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df['YearMonth'] = self.df['Date'].dt.to_period('M').astype(str)
            self.df['Month'] = self.df['Date'].dt.strftime('%b %Y')
            self.df['DayOfWeek'] = self.df['Date'].dt.day_name()
        
        # Handle numerical types
        num_cols = ['Unit_Price', 'Units_Sold', 'Total_Revenue', 'Profit', 'Customer_Rating']
        for col in num_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Fill missing values if any
        self.df.fillna({
            'Total_Revenue': 0.0,
            'Profit': 0.0,
            'Units_Sold': 0,
            'Customer_Rating': 3.5
        }, inplace=True)
        
        # Calculate profit margin percentage if not existing
        if 'Profit' in self.df.columns and 'Total_Revenue' in self.df.columns:
            self.df['Profit_Margin_%'] = np.where(
                self.df['Total_Revenue'] > 0,
                (self.df['Profit'] / self.df['Total_Revenue']) * 100,
                0.0
            )

        return self.df

    def get_summary_stats(self) -> dict:
        """
        Returns high-level KPI summary statistics.
        """
        if self.df is None or self.df.empty:
            return {}
        
        return {
            "total_transactions": len(self.df),
            "total_revenue": float(self.df['Total_Revenue'].sum()),
            "total_profit": float(self.df['Profit'].sum()),
            "total_units_sold": int(self.df['Units_Sold'].sum()),
            "avg_order_value": float(self.df['Total_Revenue'].mean()),
            "avg_customer_rating": float(self.df['Customer_Rating'].mean()),
            "num_products": self.df['Product_Name'].nunique() if 'Product_Name' in self.df.columns else 0,
            "num_customers": self.df['Customer_ID'].nunique() if 'Customer_ID' in self.df.columns else 0
        }
