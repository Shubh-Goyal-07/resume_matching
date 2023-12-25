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


pro1 = "Developed a website for IIT Jodhpur's Literature Society featuring user profile management, in-site notifications, library management, a reader's section for written works upload, and moderator/admin access for content approval."

# pro2 = "Developed pipeline for bias/toxicity detection in outputs of large language models and integrated GPT-2, BERT and other language models for testing. Developed user-friendly UI (ReactJS) and backend (Django Rest Framework) for integration of detectors and providing an intuitive platform for using the developed pipeline."
pro2 = "Established bias and toxicity detection pipeline for large language models, incorporating GPT-2, BERT, and others. Designed a user-friendly UI (ReactJS) and backend (Django Rest Framework) to seamlessly integrate detectors, providing an intuitive platform for pipeline usage."

# pro3 = "Performed data analysis and outlier detection on a dataset of accelerometer and gyroscope readings. Ideated and executed a pipeline for performing cleaning and detecting the activity type throught the dataset."

pro3 = "Conducted data analysis and outlier detection on accelerometer and gyroscope readings. Executed a pipeline for cleaning and activity type detection in the dataset."

pro4 = "Designed a Food Delivery and Ordering App for Shamiyana Cafe with objectives to streamline the ordering process and enhance cafe management through a web portal, delivering an efficient online food ordering experience for customers."

# jdk1 = "Data Analysis and Preprocessing: Collect and preprocess large-scale text corpora to train and fine-tune language models. Conduct data analysis to identify patterns, trends, and insights that can inform model development and improvement. \nPrototypes and do proof of concepts (PoC) in one or all the following areas: LLM, NLP, DL (Deep Learning), ML (Machine Learning), object detection/classification, tracking, etc"

# jdk1 = "Gather and preprocess large text datasets for LLM, NLP, DL, ML models. Analyze data to identify patterns. Prototype and perform PoCs in object detection, tracking, and more."

# jdk1 = "Hiring front-end developers with a focus on design, responsible for web design alignment, user experience optimization, and maintaining brand consistency. Collaboration with back-end developers, graphic designers, and user experience designers is essential."
# jdk1 = "Successfully implemented end-to-end development using the MERN stack, with a focus on integrating Large Language Models (LLM). Executed data preprocessing and analysis, specializing in LLM, NLP, DL, ML, object detection, and tracking. Played a pivotal role in code maintenance, scalability, and feature development for web/mobile applications"
jdk1 = "A cold breeze sweeps through the western hills as darkness falls."
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
