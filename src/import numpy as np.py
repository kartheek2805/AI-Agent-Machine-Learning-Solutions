import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


class RecommendationAgent:

    def __init__(self, ratings_df: pd.DataFrame):
        self.ratings_df = ratings_df
        self.user_item_matrix = None
        self.item_similarity = None
        self._fit()

    # ------------------------------------------------
    # CREATE MATRIX + ITEM SIMILARITY
    # ------------------------------------------------
    def _fit(self):

        # Create User-Item Matrix
        self.user_item_matrix = self.ratings_df.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating"
        ).fillna(0)

        # Calculate cosine similarity between items
        sim = cosine_similarity(self.user_item_matrix.T)

        self.item_similarity = pd.DataFrame(
            sim,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )

    # ------------------------------------------------
    # PREDICT RATING
    # ------------------------------------------------
    def predict_rating(self, user_id, item_id, k=3):

        if item_id not in self.item_similarity.columns:
            return np.nan

        if user_id not in self.user_item_matrix.index:
            return np.nan

        user_ratings = self.user_item_matrix.loc[user_id]

        # Items already rated by user
        rated_items = user_ratings[user_ratings > 0]

        if rated_items.empty:
            return np.nan

        # Similarity between target item and rated items
        sims = self.item_similarity.loc[
            item_id,
            rated_items.index
        ]

        # Select top k similar items
        top_k = sims.sort_values(
            ascending=False
        ).head(k)

        if top_k.sum() == 0:
            return np.nan

        # Weighted average
        weighted_sum = (
            top_k * rated_items[top_k.index]
        ).sum()

        predicted_rating = weighted_sum / top_k.sum()

        return round(predicted_rating, 2)

    # ------------------------------------------------
    # RECOMMEND ITEMS
    # ------------------------------------------------
    def recommend(self, user_id, n=5):

        if user_id not in self.user_item_matrix.index:
            return []

        user_ratings = self.user_item_matrix.loc[user_id]

        # Find items user has NOT rated
        unrated_items = user_ratings[
            user_ratings == 0
        ].index

        predictions = {}

        for item in unrated_items:

            prediction = self.predict_rating(
                user_id,
                item
            )

            if not np.isnan(prediction):
                predictions[item] = prediction

        # Sort recommendations
        ranked = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:n]

    # ------------------------------------------------
    # GRAPH 1: USER ITEM MATRIX
    # ------------------------------------------------
    def plot_rating_matrix(self):

        plt.figure(figsize=(8, 5))

        sns.heatmap(
            self.user_item_matrix,
            annot=True,
            cmap="Blues",
            linewidths=0.5
        )

        plt.title("User-Item Rating Matrix")
        plt.xlabel("Item ID")
        plt.ylabel("User ID")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------
    # GRAPH 2: ITEM SIMILARITY
    # ------------------------------------------------
    def plot_similarity(self):

        plt.figure(figsize=(8, 6))

        sns.heatmap(
            self.item_similarity,
            annot=True,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            linewidths=0.5
        )

        plt.title("Item Similarity Heatmap")
        plt.xlabel("Item ID")
        plt.ylabel("Item ID")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------
    # GRAPH 3: RECOMMENDATION GRAPH
    # ------------------------------------------------
    def plot_recommendations(self, user_id, n=5):

        recommendations = self.recommend(
            user_id,
            n
        )

        if not recommendations:
            print("No recommendations available.")
            return

        items = [
            str(item)
            for item, rating in recommendations
        ]

        predicted_ratings = [
            rating
            for item, rating in recommendations
        ]

        plt.figure(figsize=(8, 5))

        bars = plt.bar(
            items,
            predicted_ratings
        )

        plt.title(
            f"Top Recommendations for User {user_id}"
        )

        plt.xlabel("Item ID")
        plt.ylabel("Predicted Rating")

        plt.ylim(0, 5)

        # Display rating on top of bars
        for bar, rating in zip(
            bars,
            predicted_ratings
        ):

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                str(rating),
                ha="center"
            )

        plt.tight_layout()
        plt.show()


# ====================================================
# DATA
# ====================================================

data = pd.DataFrame({

    "user_id": [
        1, 1, 1,
        2, 2, 2,
        3, 3
    ],

    "item_id": [
        101, 102, 103,
        101, 103, 104,
        102, 104
    ],

    "rating": [
        5, 3, 4,
        4, 5, 2,
        5, 3
    ]
})


# ====================================================
# CREATE RECOMMENDATION AGENT
# ====================================================

agent = RecommendationAgent(data)


# ====================================================
# PRINT USER-ITEM MATRIX
# ====================================================

print("\nUSER-ITEM MATRIX")
print("================")
print(agent.user_item_matrix)


# ====================================================
# PRINT ITEM SIMILARITY
# ====================================================

print("\nITEM SIMILARITY")
print("================")
print(agent.item_similarity.round(2))


# ====================================================
# GET RECOMMENDATIONS
# ====================================================

user_id = 1

recommendations = agent.recommend(
    user_id=user_id,
    n=3
)

print("\nRECOMMENDATIONS")
print("================")

for item, rating in recommendations:

    print(
        f"Item {item} → Predicted Rating: {rating}"
    )


# ====================================================
# SHOW GRAPHS
# ====================================================

# Graph 1
agent.plot_rating_matrix()

# Graph 2
agent.plot_similarity()

# Graph 3
agent.plot_recommendations(
    user_id=1,
    n=3
)