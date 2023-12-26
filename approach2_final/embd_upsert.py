from langchain.chains import LLMChain
from langchain.llms import OpenAI as OpenAI_llm
from openai import OpenAI
from langchain.prompts import PromptTemplate

import openai
import os

import pinecone

class Upsert_model():
    def __init__(self, data):
        self.data = data


    def __create_jdk_prompt(self):
        template = """Job Title: {title}
        Job Description: {description}
        
        Combine the above information to create a concise and more precise description of the job. Remove irrelevant information from the job description and only include the important information. Convert the job description into a project description."""

        self.prompt = PromptTemplate(template=template, input_variables=["title", "description"])

        return


    def __create_candidate_prompt(self):
        template = """Project Title: {title}
        Project Description: {description}
        
        Combine the above information to create a concise and more precise description of the project. Remove irrelevant information from the project description and only include the important information."""

        self.prompt = PromptTemplate(template=template, input_variables=["title", "description"])

        return


    def __get_final_description(self, title, description, skills):
        llm = OpenAI_llm()
        llm_chain = LLMChain(prompt=self.prompt, llm=llm)

        final_description = llm_chain.run(title=title, description=description)

        skills = ", ".join(skills)
        final_description = f"{final_description}. Skills used includes {skills}."

        return final_description
    

    def __upsert_to_database(self, namespace, embeddings_vector):
        
        index = pinecone.Index("willings")
        
        index.upsert(
            vectors=embeddings_vector,
            namespace=namespace
        )

        return 1


    def __get_embeddings(self, final_description):
        client = OpenAI()
        embeddings = client.embeddings.create(input = [final_description], model="text-embedding-ada-002").data[0].embedding

        return embeddings


    def add_jdk(self):
        jdk_id = self.data['id']
        title = self.data['title']
        description = self.data['description']
        skills = self.data['skills']

        self.__create_jdk_prompt()
        final_description = self.__get_final_description(title, description, skills)
        embeddings = self.__get_embeddings(final_description)

        vector = [{"id": str(jdk_id), "values": embeddings}]

        self.__upsert_to_database("jdks", vector)

        return
    

    def add_candidate(self):
        candidate_id = self.data['id']
        projects = self.data['projects']

        vector = []
        self.__create_candidate_prompt()
        for project in projects:
            title = project['name']
            description = project['description']
            skills = project['skills']

            final_description = self.__get_final_description(title, description, skills)
            embeddings = self.__get_embeddings(final_description)

            vector.append({"id": title, "values": embeddings})

        namespace = f"candidate_{candidate_id}"
        self.__upsert_to_database(namespace, vector)

        return


def upsert_to_database(category, data):
    # Load .env file
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv())
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    pinecone.init(      
        api_key=os.environ.get('PINECONE_API_KEY'),
        environment='gcp-starter'
    )


    upsert_model = Upsert_model(data)
    
    if category == "candidate":
        upsert_model.add_candidate()
    else:
        upsert_model.add_jdk()