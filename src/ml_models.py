import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity


class SalesForecaster:
    """
    Predictive Machine Learning model for Sales & Revenue Forecasting.
    """
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.label_encoders = {}
        self.feature_columns = []
        self.metrics = {}

    def prepare_features(self, df: pd.DataFrame):
        df_feat = df.copy()
        if 'Date' in df_feat.columns:
            df_feat['Year'] = pd.to_datetime(df_feat['Date']).dt.year
            df_feat['Month'] = pd.to_datetime(df_feat['Date']).dt.month
            df_feat['Day'] = pd.to_datetime(df_feat['Date']).dt.day
            df_feat['DayOfWeek'] = pd.to_datetime(df_feat['Date']).dt.dayofweek

        cat_cols = ['Category', 'Region', 'Product_ID']
        for col in cat_cols:
            if col in df_feat.columns:
                le = LabelEncoder()
                df_feat[col + '_Enc'] = le.fit_transform(df_feat[col].astype(str))
                self.label_encoders[col] = le

        self.feature_columns = [c for c in [
            'Unit_Price', 'Units_Sold', 'Year', 'Month', 'Day', 'DayOfWeek',
            'Category_Enc', 'Region_Enc', 'Product_ID_Enc'
        ] if c in df_feat.columns]

        return df_feat

    def train(self, df: pd.DataFrame, target_col: str = 'Total_Revenue'):
        df_feat = self.prepare_features(df)
        X = df_feat[self.feature_columns]
        y = df_feat[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)

        self.metrics = {
            'r2': round(r2_score(y_test, preds), 4),
            'mae': round(mean_absolute_error(y_test, preds), 2),
            'rmse': round(np.sqrt(mean_squared_error(y_test, preds)), 2)
        }

        # Feature Importance
        importances = self.model.feature_importances_
        feature_imp_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        return self.metrics, feature_imp_df, X_test, y_test, preds

    def predict_future_sale(self, unit_price: float, units_sold: int, category: str = 'AI Solutions', region: str = 'North America'):
        # Construct sample input row
        feat = {
            'Unit_Price': unit_price,
            'Units_Sold': units_sold,
            'Year': 2026,
            'Month': 10,
            'Day': 15,
            'DayOfWeek': 2,
            'Category_Enc': 0,
            'Region_Enc': 0,
            'Product_ID_Enc': 0
        }
        input_df = pd.DataFrame([feat])[self.feature_columns]
        predicted_val = self.model.predict(input_df)[0]
        return max(0.0, float(round(predicted_val, 2)))


class RecommendationAgent:
    """
    Collaborative Filtering Recommendation System for Products.
    """
    def __init__(self):
        self.user_item_matrix = None
        self.item_similarity = None

    def fit(self, df: pd.DataFrame):
        # Create User-Item matrix based on Customer_Rating or Units_Sold
        if 'Customer_ID' not in df.columns or 'Product_Name' not in df.columns:
            return
        
        self.user_item_matrix = df.pivot_table(
            index='Customer_ID',
            columns='Product_Name',
            values='Customer_Rating',
            aggfunc='mean'
        ).fillna(0)

        sim_matrix = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity = pd.DataFrame(
            sim_matrix,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )

    def recommend_products_for_user(self, customer_id: str, top_n: int = 3):
        if self.user_item_matrix is None or customer_id not in self.user_item_matrix.index:
            # Fallback to top overall products
            return [("AI Analytics Suite", 4.8), ("ML Predictor Pro", 4.6), ("Cloud Agent Hub", 4.5)][:top_n]
        
        user_ratings = self.user_item_matrix.loc[customer_id]
        unrated = user_ratings[user_ratings == 0].index
        rated = user_ratings[user_ratings > 0]

        if rated.empty or len(unrated) == 0:
            return [(prod, 4.5) for prod in self.user_item_matrix.columns[:top_n]]

        predictions = {}
        for item in unrated:
            sims = self.item_similarity.loc[item, rated.index]
            if sims.sum() > 0:
                weighted_rating = (sims * rated[sims.index]).sum() / sims.sum()
                predictions[item] = round(weighted_rating, 2)
            else:
                predictions[item] = 3.5

        sorted_recs = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:top_n]


class CustomerSegmenter:
    """
    K-Means Clustering for Customer Behavioral Segmentation.
    """
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()

    def segment(self, df: pd.DataFrame):
        cust_df = df.groupby('Customer_ID').agg({
            'Total_Revenue': 'sum',
            'Transaction_ID': 'count',
            'Customer_Rating': 'mean',
            'Profit': 'sum'
        }).reset_index().rename(columns={'Transaction_ID': 'Order_Frequency'})

        features = ['Total_Revenue', 'Order_Frequency', 'Customer_Rating', 'Profit']
        X_scaled = self.scaler.fit_transform(cust_df[features])

        cust_df['Cluster'] = self.kmeans.fit_predict(X_scaled)

        # Label clusters dynamically based on mean revenue
        cluster_means = cust_df.groupby('Cluster')['Total_Revenue'].mean().sort_values(ascending=False)
        rank_map = {old_c: i for i, old_c in enumerate(cluster_means.index)}
        
        cluster_names = {
            0: 'Tier 1: High-Value VIPs',
            1: 'Tier 2: Steady Growth Customers',
            2: 'Tier 3: Occasional Buyers'
        }
        
        cust_df['Segment_Name'] = cust_df['Cluster'].map(lambda c: cluster_names.get(rank_map[c], 'Standard'))

        return cust_df
