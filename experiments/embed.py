import os
from openai import OpenAI
# from config import OPENAI_API_KEY
from dotenv import load_dotenv, find_dotenv
import pinecone
# from pinecone import IndexVector


_ = load_dotenv(find_dotenv()) # read local .env file
api_key = os.environ.get('OPENAI_API_KEY')
api_key2 = os.environ.get('PINECONE_API_KEY')

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(api_key)
print(api_key2)
client = OpenAI(api_key=api_key)
# client.api_key = os.getenv("OPENAI_API_KEY")


pro1 = "A website developed for managing various activities for literature society of IIT Jodhpur. The website contains the following features: \n-User Profile Management (with in site notification service) \n-Library Management System \n-Feature to upload written works to be displayed on website (Reader's Section). \n-Moderator and Admin access profiles to approve books and contents."

pro2 = "Developed pipeline for bias/toxicity detection in outputs of large language models and integrated GPT-2, BERT and other language models for testing. Developed user-friendly UI (ReactJS) and backend (Django Rest Framework) for integration of detectors and providing an intuitive platform for using the developed pipeline."

pro3 = "Performed data analysis and outlier detection on a dataset of accelerometer and gyroscope readings. Ideated and executed a pipeline for performing cleaning and detecting the activity type throught the dataset."

pro4 = "A Food Delivery cum ordering App for Shamiyana Cafe. \nObjectives \n- To make the ordering process hassle-free and time-saving. \n- To manage the Cafe easily by a web portal. \nDeliverables \n- An App for Shamiyana Cafe where customers can orderfood online."

jdk1 = "Data Analysis and Preprocessing: Collect and preprocess large-scale text corpora to train and fine-tune language models. Conduct data analysis to identify patterns, trends, and insights that can inform model development and improvement. \nPrototypes and do proof of concepts (PoC) in one or all the following areas: LLM, NLP, DL (Deep Learning), ML (Machine Learning), object detection/classification, tracking, etc"

# response1 = client.embeddings.create(input = [pro1], model="text-embedding-ada-002").data[0].embedding
# response2 = client.embeddings.create(input = [pro2], model="text-embedding-ada-002").data[0].embedding
# response3 = client.embeddings.create(input = [pro3], model="text-embedding-ada-002").data[0].embedding
# response4 = client.embeddings.create(input = [pro4], model="text-embedding-ada-002").data[0].embedding
response5 = client.embeddings.create(input = [jdk1], model="text-embedding-ada-002").data[0].embedding
print(response5)


pinecone.init(      
	api_key=api_key,
	environment='gcp-starter'
)      
index = pinecone.Index('res-match')

# vecs = [
    # IndexVector(id="id1", values=response1),
    # IndexVector(id="id2", values=response2),
    # IndexVector(id="id3", values=response3),
#     # IndexVector(id="id4", values=response4),
# ]

# index.upsert(vecs)

# query_vec = response5
# results = index.query(queries=[query_vec], top_k=10)

# for match in results:
    # print(f'ID: {match.id} Score: {match.score:.4f}')
