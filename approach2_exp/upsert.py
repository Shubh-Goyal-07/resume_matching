from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv, find_dotenv

import openai
import os

import pinecone

from datetime import datetime

import json


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

        final_description = f"{final_description} Skills used includes {skills}."

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


    def __get_jdk_final_description(self, title, description, skills):
        description = f"The job title is {title}, and the description is as follows: {description}. The skills required for the job are {skills}."

        return description


    def add_jdk(self):
        jdk_id = self.data['id']
        title = self.data['title']
        description = self.data['description']
        skills = self.data['skills']
        skills = ", ".join(skills)

        self.__create_jdk_prompt()
        final_description = self.__get_final_description(title, description, skills)

        embeddings = self.__get_embeddings([final_description])

        description_embeddings = embeddings[0]

        vector = [{"id": str(jdk_id), "values": description_embeddings}]

        self.__upsert_to_database("jdks", vector)
        description = self.__get_jdk_final_description(title, final_description, skills)

        return description


    def __get_cand_combined_desc(self, titles, final_descriptions, skill_info):
        all_project_desc = ""

        for title, description, skills in zip(titles, final_descriptions, skill_info):
            all_project_desc += f"The project is titled '{title}'. {description} The project uses {skills}. "

        return all_project_desc
    

    def __get_experience(self, start_date, end_date):
        date_format = "%d/%m/%Y"

        end_date_obj = datetime.strptime(end_date, date_format).date()
        start_date_obj = datetime.strptime(start_date, date_format).date()
        
        exp_diff = end_date_obj - start_date_obj
        
        # Capping experience to be not more than 6 months
        experience = min(exp_diff.days, 180) / 15 
        experience = int(round(experience, 0))

        return experience


    def __get_personality_score(self):
        client = openai.OpenAI()
        answers = self.data['compRecruitScreeningAnswers']
        answers.update(self.data['compRecruitQuestionnaireAnswers'])
        answers = str(answers)

        personality_prompt = """You are an agent that judges an applicant's personality and his/her willingness to go to Japan to work for the company. 
        
        You have to judge the willingness of the applicant to go to Japan based on his/her answers to the following set of questions:
        1. The reason why you want to come to Japan
        2. The career plan you want
        3. In which country do you want to work after graduation?
        4. Are you likely to be adaptable to other cultures?
        5. Instead of English-speaking countries like the U.S., the U.K. and Singapore, why are you interested in working in Japan?
        6. What are your expectations from the company?
        7. What would you like to accomplish during the internship with the company?

        In addition to the willingness, you also have to judge the personality of the applicant based on his/her answers to the following set of questions:
        1. Your strengths and characteristics
        2. Weaknesses or areas where they would like to improve
        3. Steps they are taking or plan to take to address these areas
        4. Example of a challenge or setback you have faced and how they overcame it
        5. Lessons learned from that experience
        6. Do you have any unique background that differentiate you from others?
        7. Do you have any specialty?

        You will be given the answers to all the questions in a JSON format. You have to give a single score based on the applicant's personality and his/her willingness to go to Japan.

        You have to return the output in the following JSON format:
        {
            score: <GIVE A SCORE OUT OF 5 HERE>,
        }
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an agent that judges an applicant's personality and his/her willingness to go to Japan to work for the company."},
                {"role": "user", "content": personality_prompt + '\n' + "Here are the answers:\n" + str(answers)},
            ]
        )

        score = json.loads(response.choices[0].message.content)['score']
        return score


    def add_candidate(self, save_gen_desc_only = False):
        candidate_id = self.data['id']
        projects = self.data['projects']

        self.__create_candidate_prompt()
        
        titles = []
        actual_titles = []
        final_descriptions = []
        skill_info = []

        for project in projects:
            title = project['title']
            description = project['description']
            skills = project['skills']
            skills = ", ".join(skills)

            end_date = project['endDate']
            start_date = project['startDate']
            experience = self.__get_experience(start_date, end_date)

            titles.append(f"{title}__{experience}")
            actual_titles.append(f"{title}")

            final_description = self.__get_final_description(title, description, skills)
            final_descriptions.append(final_description)

            skill_info.append(skills)

        all_projects_desc = self.__get_cand_combined_desc(actual_titles, final_descriptions, skill_info)
        

        if save_gen_desc_only:
            embeddings = self.__get_embeddings([all_projects_desc])
        else:
            final_descriptions.append(all_projects_desc)
            embeddings = self.__get_embeddings(final_descriptions)
            namespace = f"candidate_{candidate_id}"
            description_vector = [{"id": title, "values": embeddings[i]} for i, title in enumerate(titles)]
            self.__upsert_to_database(namespace, description_vector)


        gen_desc_vec = [{"id": f"{candidate_id}", "values": embeddings[-1]}]
        namespace_all = "all_candidates"
        self.__upsert_to_database(namespace_all, gen_desc_vec)
        
        score = self.__get_personality_score()

        return {"final_description": all_projects_desc, "score": score}


def upsert_to_database(category, data, save_gen_desc_only=False):
    # Load .env file
    _ = load_dotenv(find_dotenv())
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    pinecone.init(      
        api_key=os.environ.get('PINECONE_API_KEY'),
        environment='gcp-starter'
    )


    upsert_model = Upsert_model(data)
    
    if category == "candidate":
        description = upsert_model.add_candidate(save_gen_desc_only=save_gen_desc_only)
    else:
        description = upsert_model.add_jdk()

    return description
