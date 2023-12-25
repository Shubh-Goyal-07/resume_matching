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

# jdk1 = "Successfully implemented end-to-end development using the MERN stack, with a focus on integrating Large Language Models (LLM). Executed data preprocessing and analysis, specializing in LLM, NLP, DL, ML, object detection, and tracking. Played a pivotal role in code maintenance, scalability, and feature development for web/mobile applications. Skills required are ReactJS, Django, Python, JavaScript, Pytorch."
jdk1 = "Collect and preprocess large-scale text corpora to train and fine-tune language models. Conduct data analysis to identify patterns, trends, and insights that can inform model development and improvement. \nPrototypes and do proof of concepts (PoC) in one or all the following areas: LLM, NLP, DL (Deep Learning), ML (Machine Learning), object detection/classification, tracking, etc. Skills required are Python, Pytorch, Tensorflow, Sklearn, Data Analysis."

print("Creating embeddings...")
jdk_embed = client.embeddings.create(input = [jdk1], model="text-embedding-ada-002").data[0].embedding

print("Embeddings created.")
print("Activating Pinecone index...")

pinecone.init(      
	api_key=api_key,
	environment='gcp-starter'
)

index = pinecone.Index("quickstart")

print("Index active")
print("Sending query...")

result = index.query(
  namespace="cand1",
  vector=jdk_embed,
  top_k=10,
  include_values=False
)

print(result)