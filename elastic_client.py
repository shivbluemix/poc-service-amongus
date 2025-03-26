from elasticsearch import Elasticsearch, helpers
from datetime import datetime, timezone
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


# Function to get the current time in ISO 8601 format with milliseconds
def get_current_time_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class ElasticSearchClient:
    def __init__(self):
        # Connect to Elasticsearch
        self.es = Elasticsearch(["http://localhost:9200"])

    def create_person_identity_index(self, index_name="personidentity"):
        mapping = {
            "mappings": {
                "properties": {
                    "full_name": {
                        "type": "text",  # Full-text search
                        "fields": {"keyword": {"type": "keyword"}},  # Exact search
                    },
                    "company_name": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "zip_code": {"type": "keyword"},  # Exact search only
                    "phone1": {"type": "keyword"},  # Exact search only
                    "email": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "role": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "permission": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                }
            }
        }
        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)
            print(f"Index '{index_name}' deleted successfully.")
        else:
            print(f"Index '{index_name}' does not exist.")

        self.es.indices.create(index=index_name, body=mapping)

    def load_person_identity(self, index_name="personidentity"):
        # Read the file content
        df = pd.read_csv("data/data_with_duplicates.csv")

        # Convert the DataFrame to a list of dictionaries
        data = df.to_dict(orient="records")

        # Prepare the data for bulk indexing
        actions = [
            {
                "_index": index_name,
                "_id": i,  # Use the row index as the document ID
                "_source": record  # The document data
            }
            for i, record in enumerate(data)
        ]

        # Bulk index the data
        helpers.bulk(self.es, actions)
        print("Data indexed successfully.")

    def search_person_identity(
        self, field,value, mode="exact_match", index_name="personidentity"
    ):
        query = None
        if mode == "exact_match":
            query = {"query": {"term": {f"{field}.keyword": value}}}
        elif mode == "full_text":
            query = {"query": {"match": {field: value}}}
        response = self.es.search(index=index_name, body=query)
        if response["hits"]["hits"]:
            return response["hits"]["hits"]
        else:
            return []
        
# es = ElasticSearchClient()
# es.create_person_identity_index()
# es.load_person_identity()