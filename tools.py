from langchain_core.tools import tool
from elastic_client import ElasticSearchClient
from clustering_client import ClusteringClient


elastic_client = ElasticSearchClient()
cluster_client = ClusteringClient()



@tool
def find_sum(x: int, y: int) -> int:
    # The docstring comment describes the capabilities of the function
    # It is used by the agent to discover the function's inputs, outputs and capabilities
    """
    This function is used to add two numbers and return their sum.
    It takes two integers as inputs and returns an integer as output.
    """
    return x + y


@tool
def find_person_identity(field,value, mode) -> dict:
    """
    This function is used to find a person's identity by their name or email.
    It takes strings as input and
    returns a list of dictionary and a string of mode as output, there is a chance that the list is empty.
    Return the raw dictionary **exactly as they are**.
    """
    result = elastic_client.search_person_identity(field,value, mode)
    return result, mode


@tool
def fetch_top_k_duplicate(k):
    """
    This function is used fetch top k duplicate clusters
    It takes integer k as input and returns a list of dictionaries as the output, one fore each cluster.
    There is a chance that the list is empty.
    Return the raw result **exactly as they are**
    """
    result = cluster_client.get_largest_k_cluster(k)
    return result

@tool
def save_human_feedback(feedback, confidence_level, cluster_id):
    """
    This function is used to save human feedback for the duplication clusters
    It takes a string feedback and a string cluster_id as input and returns a string as output.
    """
    result = cluster_client.save_human_feedback(feedback,confidence_level, cluster_id)
    return result
# temp = find_person_identity("Shivam Gupta","exact_match")
# print(temp)
