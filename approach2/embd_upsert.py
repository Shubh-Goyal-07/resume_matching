from typing import Any
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv, find_dotenv

import openai
import os

import pinecone

from datetime import datetime

class Upsert_model():
    def __init__(self, data):
        self.data = data


    def __create_jdk_prompt(self):
        template = """Job Title: {title}
        Job Description: {description}
        
        Combine the above information to create a concise and more precise description of the job. Remove irrelevant information from the job description and only include the important information. Convert the job description into a project description. Wrap it in two sentences."""

        self.prompt = PromptTemplate(template=template, input_variables=["title", "description"])

        return


    def __create_candidate_prompt(self):
        template = """Project Title: {title}
        Project Description: {description}
        
        Combine the above information to create a concise and more precise description of the project. Remove irrelevant information from the project description and only include the important information. Wrap it in two sentences."""

        self.prompt = PromptTemplate(template=template, input_variables=["title", "description"])

        return


    def __get_final_description(self, title, description, skills):
        llm = OpenAI()
        llm_chain = LLMChain(prompt=self.prompt, llm=llm)

        final_description = llm_chain.run(title=title, description=description)

        final_description = f"{final_description}. Skills used includes {skills}."

        return final_description
    

    def __upsert_to_database(self, namespace, embeddings_vector):
        
        index = pinecone.Index("willings")
        
        index.upsert(
            vectors=embeddings_vector,
            namespace=namespace
        )

        return 1


    def __get_embeddings(self, description_list):
        embedding_model = OpenAIEmbeddings()
        embeddings = embedding_model.embed_documents(description_list)

        return embeddings


    def add_jdk(self):
        jdk_id = self.data['id']
        title = self.data['title']
        description = self.data['description']
        skills = self.data['skills']
        skills = ", ".join(skills)

        self.__create_jdk_prompt()
        final_description = self.__get_final_description(title, description, skills)

        skills_description = f"{title} is a job that uses {skills}."
        embeddings = self.__get_embeddings([final_description, skills_description])

        description_embeddings = embeddings[0]
        skill_embeddings = embeddings[1]

        vector = [{"id": str(jdk_id), "values": description_embeddings}]
        skill_vector = [{"id": str(jdk_id), "values": skill_embeddings}]

        self.__upsert_to_database("jdks", vector)
        self.__upsert_to_database("jdks_skills", skill_vector)

        return
    

    def __get_experience(self, start_date, end_date):
        date_format = "%d/%m/%Y"

        end_date_obj = datetime.strptime(end_date, date_format).date()
        start_date_obj = datetime.strptime(start_date, date_format).date()
        
        exp_diff = end_date_obj - start_date_obj
        
        # Capping experience to be not more than 6 months
        experience = min(exp_diff.days, 180) / 15 
        experience = int(round(experience, 0))

        return experience
    

    def add_candidate(self):
        candidate_id = self.data['id']
        projects = self.data['projects']

        self.__create_candidate_prompt()
        
        num_projects = len(projects)

        titles = []
        skill_titles = []
        final_descriptions = []
        skill_descriptions = []

        date_format = "%d/%m/%Y"

        for project in projects:
            title = project['title']
            description = project['description']
            skills = project['skills']
            skills = ", ".join(skills)

            end_date = project['endDate']
            start_date = project['startDate']
            experience = self.__get_experience(start_date, end_date)

            titles.append(f"{title}__{experience}")
            skill_titles.append(f"{title}")

            final_description = self.__get_final_description(title, description, skills)
            final_descriptions.append(final_description)

            skill_description = f"{title} is a project that uses {skills}."
            skill_descriptions.append(skill_description)


        final_descriptions.extend(skill_descriptions)
        embeddings = self.__get_embeddings(final_descriptions)


        description_vector = [{"id": title, "values": embeddings[i]} for i, title in enumerate(titles)]
        skill_vector = [{"id": title, "values": embeddings[i]} for i, title in enumerate(skill_titles, start=num_projects)]

        namespace = f"candidate_{candidate_id}"
        self.__upsert_to_database(namespace, description_vector)

        skill_namespace = f"candidate_{candidate_id}_skills"
        self.__upsert_to_database(skill_namespace, skill_vector)

        return


def upsert_to_database(category, data):
    # Load .env file
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