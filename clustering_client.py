from sklearn.cluster import DBSCAN
from langchain_openai import AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

class ClusteringClient:
    def __init__(self):
        self.model = AzureOpenAIEmbeddings(
            model="text-embedding-3-large",
        )
        self.cache = set()
    def load_data(self):
        self.df = pd.read_csv("data/data_with_duplicates.csv")
        self.df["full_name_with_phone"] = self.df["full_name"] + " " + self.df["phone1"]

        # Generate embeddings for the concatenated column
        embeddings = self.model.embed_documents(self.df["full_name_with_phone"].tolist())
        # reducer = umap.UMAP(metric="cosine")
        # reduced = reducer.fit_transform(embeddings)
        ml_model = DBSCAN(eps=0.1,min_samples=1,metric="euclidean")
        clusters = ml_model.fit_predict(embeddings)
        self.df["cluster"] = clusters
        self.df.to_csv("data/clustered_data.csv",index=False)

    def get_largest_k_cluster(self,k):
        self.df = pd.read_csv("data/clustered_data.csv")
        cluster_counts = self.df["cluster"].value_counts()
        # Get the top K cluster labels
        largest_clusters = [
            cluster for cluster in cluster_counts.index if cluster not in self.cache
        ][:k]

        # Add the selected clusters to the cache
        self.cache.update(largest_clusters)

        # Create a list of DataFrames, one for each cluster
        cluster_dataframes = [
            self.df[self.df["cluster"] == cluster] for cluster in largest_clusters
        ]
        return [dataframe.to_dict(orient="records") for dataframe in cluster_dataframes]
    
    def save_human_feedback(self, feedback, confidence_level,cluster_id):
        # Save the feedback to a file
        with open("data/feedback.txt", "a") as f:
            f.write(f"cluster:{cluster_id} feedback:{feedback} confidence_level:{confidence_level}\n")
        return "Feedback saved successfully"
    

# cluster_client = ClusteringClient()
# # # cluster_client.load_data()
# temp = cluster_client.get_largest_k_cluster(2)
# print(temp[0].columns)
# print("here")
# print(cluster_client.get_largest_k_cluster(5))