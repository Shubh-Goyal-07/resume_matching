import os
from dotenv import load_dotenv, find_dotenv
import pinecone
from openai import OpenAI
# from pinecone import IndexVector


_ = load_dotenv(find_dotenv()) # read local .env file

api_key = os.environ.get('PINECONE_API_KEY')
api_key_openai = os.environ.get('OPENAI_API_KEY')


print("Setting up OpenAI client...")
client = OpenAI(api_key=api_key_openai)

pro1 = "Developed a website for IIT Jodhpur's Literature Society featuring user profile management, in-site notifications, library management, a reader's section for written works upload, and moderator/admin access for content approval. Skills used includes ReactJS, Django, Python, JavaScript."
pro2 = "Established bias and toxicity detection pipeline for large language models, incorporating GPT-2, BERT, and others. Designed a user-friendly UI (ReactJS) and backend (Django Rest Framework) to seamlessly integrate detectors, providing an intuitive platform for pipeline usage. Skills used includes ReactJS, Django, Python, JavaScript."
pro3 = "Conducted data analysis and outlier detection on accelerometer and gyroscope readings. Executed a pipeline for cleaning and activity type detection in the dataset. Skills used includes Data Analysis, PyTorch, Tensorflow, Sklearn, Python."
pro4 = "Designed a Food Delivery and Ordering App for Shamiyana Cafe with objectives to streamline the ordering process and enhance cafe management through a web portal, delivering an efficient online food ordering experience for customers. Skills used includes Flutter, Dart, Firebase."
jdk1 = "Successfully implemented end-to-end development using the MERN stack, with a focus on integrating Large Language Models (LLM). Executed data preprocessing and analysis, specializing in LLM, NLP, DL, ML, object detection, and tracking. Played a pivotal role in code maintenance, scalability, and feature development for web/mobile applications"

print("Creating embeddings...")
response1 = client.embeddings.create(input = [pro1], model="text-embedding-ada-002").data[0].embedding
response2 = client.embeddings.create(input = [pro2], model="text-embedding-ada-002").data[0].embedding
response3 = client.embeddings.create(input = [pro3], model="text-embedding-ada-002").data[0].embedding
response4 = client.embeddings.create(input = [pro4], model="text-embedding-ada-002").data[0].embedding

print("Embeddings created.")
print("Creating Pinecone index...")

pinecone.init(      
	api_key=api_key,
	environment='gcp-starter'
)

pinecone.create_index("quickstart", dimension=1536, metric="cosine")
print(pinecone.describe_index("quickstart"))

index = pinecone.Index("quickstart")

print("Index created.")

print("Upserting vectors...")

index.upsert(
  vectors=[
    {"id": "pro1", "values": response1},
    {"id": "pro2", "values": response2},
    {"id": "pro3", "values": response3},
    {"id": "pro4", "values": response4}
  ],
  namespace="cand1"
)

print("Vectors upserted.")

print(index.describe_index_stats())