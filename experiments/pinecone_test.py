import pinecone
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ['PINECONE_API_KEY']

pinecone.init(      
	api_key=api_key,
	environment='gcp-starter'
)      
index = pinecone.Index('res-match')

vecs = [
    pinecone.IndexVector(id="id1", values='PUT_PROJECT_EMBEDDING_AS_A_LIST_HERE'),
    pinecone.IndexVector(id="id2", values='PUT_PROJECT_EMBEDDING_AS_A_LIST_HERE')
]

index.upsert(vecs)

query_vec = 'PUT_JDK_EMBEDDING_AS_A_LIST_HERE'
results = index.query(queries=[query_vec], top_k=10)

for match in results:
    print(f'ID: {match.id} Score: {match.score:.4f}')