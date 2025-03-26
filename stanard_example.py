import os

import base64

import faiss
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()
# endpoint = os.getenv(
#     "ENDPOINT_URL", "https://innovate2025te5669273486.openai.azure.com/"
# )

# deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini-2")

# subscription_key = os.getenv(
#     "AZURE_OPENAI_API_KEY",
#     "4vdTuPSyyJV0ZA61FqXpH6EArgsuB0kG28KpW8e3fp3EYdb0FqZzJQQJ99BCACHYHv6XJ3w3AAAAACOG04IK",
# )


# # Initialize Azure OpenAI Service client with key-based authentication

# client = AzureOpenAI(
#     azure_endpoint=endpoint,
#     api_key=subscription_key,
#     api_version="2024-05-01-preview",
# )


# # IMAGE_PATH = "YOUR_IMAGE_PATH"

# # encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')


# # Prepare the chat prompt

# chat_prompt = [
#     {
#         "role": "system",
#         "content": [
#             {
#                 "type": "text",
#                 "text": "You are an AI assistant that helps people find information.",
#             }
#         ],
#     },
#     {"role": "user", "content": [{"type": "text", "text": "what are you?"}]},
#     {
#         "role": "assistant",
#         "content": [
#             {
#                 "type": "text",
#                 "text": "I am an AI assistant designed to help you find information, answer questions, and provide assistance on a wide range of topics. My training includes a variety of subjects, allowing me to offer support in areas such as general knowledge, technology, science, history, and more. If you have any questions or need information, feel free to ask!",
#             }
#         ],
#     },
# ]


# # Include speech result if speech is enabled

# messages = chat_prompt


# # Generate the completion

# completion = client.chat.completions.create(
#     model=deployment,
#     messages=messages,
#     max_tokens=800,
#     temperature=0.7,
#     top_p=0.95,
#     frequency_penalty=0,
#     presence_penalty=0,
#     stop=None,
#     stream=False,
# )


# print(completion.to_json())


import os

from azure.core.credentials import AzureKeyCredential
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import json
from langchain.vectorstores import FAISS
from datetime import datetime, timezone
from langchain.docstore.in_memory import InMemoryDocstore
from sklearn.cluster import DBSCAN
import numpy as np
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

embeddings_model = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
)
model = SentenceTransformer("all-MiniLM-L6-v2")
# index = "temp_index"

# # Create an in-memory docstore
# docstore = InMemoryDocstore()

# # Create a mapping from index to docstore ID
# index_to_docstore_id = {}


# # Initialize the FAISS vector store with the required arguments
# vector_store = FAISS(
#     embedding_function=embeddings,
#     index=index,
#     docstore=docstore,
#     index_to_docstore_id=index_to_docstore_id
# )
def get_current_time_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# embedding_model.get_embedding("Hello, how are you?")
# result = embeddings.embed_documents(["Hello, how are you?","Hello, how are you?"])
# print(len(result))
# print(type(result))
# from langchain_core.vectorstores import InMemoryVectorStore

# text = "LangChain is the framework for building context-aware reasoning applications"
with open("data/personidentity.identity_dev.json", "r") as file:
    data = json.load(file)
# Preprocess the data
# documents = [
#     {"id": "1", "text": "This is the first document."},
#     {"id": "2", "text": "This is the second document."},
#     {"id": "3", "text": "This is the third document."}
# ]
documents = []
id = 1
for item in data:
    if id > 1000:
        break
    # Use _id.$oid as the unique identifier
    item["_id"] = item["_id"]["$oid"]  # Convert ObjectId to string
    # Convert MongoDB date fields to ISO 8601 strings
    if "createdDate" in item:
        item["createdDate"] = item["createdDate"]["$date"]
    else:
        item["createdDate"] = get_current_time_iso()
    if "lastModifiedDate" in item:
        item["lastModifiedDate"] = item["lastModifiedDate"]["$date"]
    else:
        item["lastModifiedDate"] = get_current_time_iso()
    phone_number = ""
    if item.get("phoneNumbers") and len(item["phoneNumbers"]) > 0:
        phone_number = item["phoneNumbers"][0].get("number", "")
    type = ""
    if item.get("type"):
        type = item["type"]
    # Prepare the text for embedding
    text = f"fullName:{item['fullName']} email:{item['email']} region:{item['regionId']} type:{type} phoneNumber:{phone_number}"
    documents.append({"id": str(id), "text": text})
    id += 1

metadata = [{"id": doc["id"], "text": doc["text"]} for doc in documents]
texts = [doc["text"] for doc in documents]
# embeddings = embeddings_model.embed_documents(texts)
# embeddings = np.array(embeddings).astype('float32')
# d = embeddings.shape[1]  # dimension
# index = faiss.IndexFlatL2(d)  # L2 distance
# index.add(embeddings)  # add vectors to the index

embeddings = model.encode(texts)
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
d = embeddings.shape[1]  # dimension
index = faiss.IndexFlatL2(d)  # L2 distance
index.add(embeddings)  # add vectors to the index

# Save index to file
faiss.write_index(index, "text_index.faiss")
index = faiss.read_index("text_index.faiss")
# docstore = InMemoryDocstore()
documents = {i: Document(page_content=text) for i, text in enumerate(texts)}

# Initialize InMemoryDocstore
docstore = InMemoryDocstore(documents)
index_to_docstore_id = {i: i for i in range(len(texts))}
# Create LangChain FAISS vector store
vector_store = FAISS(
    embedding_function=embeddings_model,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
)

# Define retriever
retriever = vector_store.as_retriever()

# Retrieve top-k most similar embeddings
query = "find me users which has same name but multiple different email"
# query_embedding = embeddings_model.embed_query([query]).astype('float32')
# results = retriever.retrieve(query_embedding, k=5)  # Request top 5 results
results = retriever.invoke(query, top_k=5)

for result in results:
    print(result)

# vectorstore = InMemoryVectorStore.from_texts(
#     [doc["text"] for doc in documents],
#     embedding=embeddings,
# )
# retriever = vectorstore.as_retriever()

# retrieved_documents = retriever.invoke("find me users which has same name but multiple different email",top_k=10)

# # # show the retrieved document's content
# # print(retrieved_documents[0].page_content)
# print(retrieved_documents)
# print(len(retrieved_documents))
# Use the vectorstore as a retriever
# retriever = vectorstore.as_retriever()
# vector_store.add_texts(
#     texts=[doc["text"] for doc in documents],  # Extract the text field
#     metadatas=[{"id": doc["id"]} for doc in documents]  # Extract the id field as metadata
# )

# Save the vector store locally
# vector_store.save_local("local_vector_store")
# vectorstore = InMemoryVectorStore.from_texts(
#     [text],
#     embedding=embeddings,
# )

# # Use the vectorstore as a retriever
# retriever = vectorstore.as_retriever()

# # Retrieve the most similar text
# retrieved_documents = retriever.invoke("What is LangChain?")

# # show the retrieved document's content
# print(retrieved_documents[0].page_content)
