from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv, find_dotenv

import openai
import os

import numpy as np

import pinecone

import time
import math

from save_scores import save_to_excel

import pandas as pd

import json

# from googletrans import Translator


class HRAssistant():
    def __init__(self, jdk_id, candidate_id_list):
        self.jdk_id = jdk_id
        self.candidate_id_list = candidate_id_list
        
        self.index = pinecone.Index("willings")


    def __fetch_jdk_embeddings(self):
        jdk_id_str = str(self.jdk_id)

        jdk_embedding = self.index.fetch(ids=[jdk_id_str], namespace="jdks").to_dict()
        jdk_embedding = jdk_embedding['vectors'][jdk_id_str]['values']

        # jdk_skill_embeddings = self.index.fetch(ids=[jdk_id_str], namespace="jdks_skills").to_dict()
        # jdk_skill_embeddings = jdk_skill_embeddings['vectors'][jdk_id_str]['values']

        return jdk_embedding


    def __fetch_candidate_scores(self, candidate_id):
        candidate_namespace = f"candidate_{candidate_id}"
        # candidate_skill_namespace = f"candidate_{candidate_id}_skills"

        cand_project_query_response = self.index.query(
                                            vector=self.jdk_embedding,
                                            namespace=candidate_namespace,
                                            top_k=10,
                                            include_values=False
                                        )['matches']
        
        # cand_skill_query_response = self.index.query(
        #                                     vector=self.jdk_skill_embeddings,
        #                                     namespace=candidate_skill_namespace,
        #                                     top_k=10,
        #                                     include_values=False
        #                                 )['matches']
        
        
        project_scores = {}
        # skill_scores = {}

        for cand_project in cand_project_query_response:
            cand_project_id = cand_project['id']
            cand_project_score = cand_project['score']
            
            if cand_project_score>=0.9:
                cand_project_score = 1
            elif cand_project_score>=0.8:
                cand_project_score += 0.05
            elif cand_project_score<=0.75:
                cand_project_score -= 0.05
            elif cand_project_score<=0.7:
                cand_project_score = 0

            if (cand_project_score):
                project_scores[cand_project_id] = round(cand_project_score, 2)


        # for cand_skill in cand_skill_query_response:
        #     cand_skill_id = cand_skill['id']
        #     cand_skill_score = cand_skill['score']

        #     skill_scores[cand_skill_id] = round(cand_skill_score, 2)

        return project_scores


    def __create_dataframe(self, project_scores):
        project_names = []
        project_scores_list = []
        # num_projects = []
        cand_ids = []
        project_experiences = []

        for candidate_id in project_scores:
            
            projects = project_scores[candidate_id]
            # num_projects.append(len(projects))
            
            for project in projects:
                cand_ids.append(candidate_id)
                project_scores_list.append(projects[project])

                name, experience = project.split("__")
                project_names.append(name)
                project_experiences.append(float(experience))

        cands_data_dict = {"id": cand_ids, "name": project_names, "project_score": project_scores_list, "experience": project_experiences}
        # print(cands_data_dict)
        cands_dataframe = pd.DataFrame(cands_data_dict)

        return cands_dataframe


    def __normalize_project_scores(self):
        
        # Step 1: Get the mean of the column
        project_score_mean = self.cands_dataframe['project_score'].mean()

        # Step 2: Subtract the mean from the column to create a new column
        self.cands_dataframe['project_devs'] = self.cands_dataframe['project_score'] - project_score_mean

        # Step 3: Apply a lambda function on the new column
        self.cands_dataframe['project_devs'] = self.cands_dataframe['project_devs'].apply(lambda x: max(round(2-math.exp(-3*x), 2), 0))

        # Step 4: Multiply the first column and the new column and store the value in the first column
        self.cands_dataframe['project_score'] = self.cands_dataframe['project_score'] * self.cands_dataframe['project_devs']

        # Step 5: Delete the new column
        self.cands_dataframe.drop('project_devs', axis=1, inplace=True)

        # Step 6: Apply a lambda function to the first column
        self.cands_dataframe['project_score'] = self.cands_dataframe['project_score'].apply(lambda x: 0 if x<0.6 else round(x*100, 2))

        return 1


    def __drop_irrelevant_projects(self):
        self.cands_dataframe = self.cands_dataframe[self.cands_dataframe['project_score']!=0]
        return 1


    def __normalize_experience_scores(self):
        column_percentiles = np.percentile(self.cands_dataframe['experience'], [75, 50, 25])

        # The result will be an array containing the 75th, 50th, and 25th percentiles
        percentile_75th = column_percentiles[0]
        percentile_50th = column_percentiles[1]
        percentile_25th = column_percentiles[2]

        self.cands_dataframe['experience'] = self.cands_dataframe['experience'].apply(lambda x: 1 if x>=percentile_75th else (0.9 if x>=percentile_50th else (0.85 if x>=percentile_25th else 0.8)))

        return 1


    def __create_final_scores_dataframe(self):
        self.cands_dataframe['final_score'] = self.cands_dataframe['project_score'] * self.cands_dataframe['experience']
        self.cands_dataframe['final_score'] = self.cands_dataframe['final_score'].apply(lambda x: min(round(x, 2), 100))
        
        self.cands_dataframe['project_count'] = self.cands_dataframe.groupby('id')['final_score'].transform('count')
        self.cands_final_score_dataframe = self.cands_dataframe.groupby('id').agg({'final_score': 'sum', 'project_count': 'first'}).reset_index()
        self.cands_final_score_dataframe['final_score'] = self.cands_final_score_dataframe['final_score'] / self.cands_final_score_dataframe['project_count']
        
        # print(self.cands_final_score_dataframe)

        return 1


    def __calc_project_count_final_normalized_scores(self):
        total_entries = len(self.cands_final_score_dataframe)
        # mode = self.cands_final_score_dataframe['project_count'].mode()[0]

        count_dataframe = pd.DataFrame(self.cands_final_score_dataframe['project_count'].value_counts())
        count_dataframe['percentage'] = (count_dataframe['count'] / total_entries) * 100

        percentage_list = [0, 0, 0, 0, 0]
        penalties = [1, 1, 1, 1, 1]
        max_percentage = 0
        for index, row in count_dataframe.iterrows():
            percentage_list[index-1] = row['percentage']
            if row['percentage']>=max_percentage:
                max_percentage = row['percentage']
                mode = index


        if mode==1:
            if percentage_list[0]>=50:
                penalties[0] = 0.95
            else:
                penalties[0] = 0.9
        
        elif mode==2:
            penalties[0] = 0.85
            percentage_345 = sum(percentage_list[2:]) 
            if percentage_345 > 0.75*percentage_list[1]:
                penalties[1] = 0.95
        
        elif mode==3:
            penalties[0] = 0.8
            percentage_45 = sum(percentage_list[3:])
            if percentage_45 > 0.75*percentage_list[2]:
                penalties[2] = 0.95
                penalties[1] = 0.9
            else:
                penalties[1] = 0.95

        elif mode==4 or mode==5:
            penalties[2] = 0.95
            penalties[1] = 0.85
            penalties[0] = 0.75  

        
        self.cands_final_score_dataframe['project_count'] = self.cands_final_score_dataframe['project_count'].apply(lambda x: penalties[x-1])
        self.cands_final_score_dataframe['final_score'] = self.cands_final_score_dataframe['final_score'] * self.cands_final_score_dataframe['project_count']
        self.cands_final_score_dataframe['final_score'] = self.cands_final_score_dataframe['final_score'].apply(lambda x: round(x, 2))
        
        # self.cands_final_score_dataframe.drop('project_count', axis=1, inplace=True)


        return


    def __retrieve_jdk_info(self):
        with open(f'../new_data/jdks/{self.jdk_id}.json') as json_file:
            data = json.load(json_file)

        jdk_title = data['title']
        jdk_description = data['description']
        jdks_skills = data['skills']

        return jdk_title, jdk_description, jdks_skills


    def __retrieve_cand_projects_info(self, candidate_id):
        with open(f'../new_data/candidates/{int(candidate_id)}.json') as json_file:
            data = json.load(json_file)

        candidate_projects = data['projects']
        count = 1

        candidate_projects_info = ""

        for project in self.cands_dataframe[self.cands_dataframe['id']==candidate_id]['name']:
            candidate_projects_info += f"{count}. {project} : {candidate_projects[project]}"
        
        return candidate_projects_info


    def __create_llm_chain(self):
        # template = """You are a reasoning agent. We have job description for a job position in the field of technology.
        # Multiple candidates applied for the job. All of them submitted their resumes and we have calculated a score that shows the aptness of the applicant for the job position.
        
        # We will give you a job description and the set of projects of the applicant alongwith the score that we calculated. You have to analyse the job description, the projects, and provide a reasoning for why the applicant has been given that score. If you think that the candidate is not suitable for the job, you can say that as well.
        
        # You have to return the output in the following format. Remember to be very brief while providing the reasoning. Try not to exceed 60 words.
        
        
        # The job title is {jdk_title}, and the description is as follows: {jdk_description}. The skills required for the job are {jdk_skills}.

        # The candidate has worked on the following projects: {candidate_projects_info}.

        # The candidate has been given a score of {candidate_score}.

        
        # Reasoning: <A VERY SUCCINT REASONING>"""

        template = """You are a reasoning agent. We have job description for a job position in the field of technology.
        Multiple candidates applied for the job. All of them submitted their resumes and we have calculated a score that shows the aptness of the applicant for the job position.
        
        We will give you a job description and the set of projects of the applicant alongwith the score that we calculated. You have to analyse the job description, the projects, and provide a reasoning for why the applicant has been given that score.

        The score is given out of 100. A candidate may get a high, low, or a moderate score. So carefully analyze the job description, the projects and then provide a reasoning as to why the applicant has a particular score. Say an applicant has a bad score then you need to justify how the applicant is not so well suited for the job based on the job description and the applicant's projects. Similarly if the applicant has a high score then you need to provide a reasoning as to why the applicant is suited for the job.
        

        The job title is {jdk_title}, and the description is as follows: {jdk_description}. The skills required for the job are {jdk_skills}.

        The candidate has worked on the following projects: {candidate_projects_info}.

        The candidate has been given a score of {candidate_score}.

        You have to return the output in the following format. Remember to be very brief while providing the reasoning. Try not to exceed 60 words.
        
        Reasoning: <A VERY SUCCINT REASONING>"""

        prompt = PromptTemplate(template=template, input_variables=["jdk_title", "jdk_description", "jdk_skills", "candidate_projects_info", "candidate_score"])

        self.llm_chain = LLMChain(prompt=prompt, llm=OpenAI())

        return


    def __add_cand_score_reasons(self):
        jdk_title, jdk_description, jdk_skills = self.__retrieve_jdk_info()
        self.__create_llm_chain()

        # translator = Translator()

        for index, row in self.cands_final_score_dataframe.iterrows():
            candidate_id = row['id']
            candidate_score = row['final_score']
            candidate_projects_info = self.__retrieve_cand_projects_info(candidate_id)

            final_reason = self.llm_chain.run(jdk_title=jdk_title, jdk_description=jdk_description, jdk_skills=jdk_skills, candidate_projects_info=candidate_projects_info, candidate_score=candidate_score)

            # final_reason_jap = translator.translate(final_reason, dest='ja')

            self.cands_final_score_dataframe.loc[index, 'reason'] = final_reason
            # self.cands_final_score_dataframe.loc[index, 'jap_reason'] = final_reason_jap

        return


    def __get_all_candidate_scores(self):
        
        project_scores_all = {}
        skill_scores_dict = {}

        for candidate_id in self.candidate_id_list:
            project_scores= self.__fetch_candidate_scores(candidate_id)
            project_scores_all[candidate_id] = project_scores
            # skill_scores_dict[candidate_id] = skill_scores

        self.cands_dataframe = self.__create_dataframe(project_scores_all)
        # print(self.cands_dataframe)
        self.__normalize_project_scores()
        # print(self.cands_dataframe)
        self.__drop_irrelevant_projects()
        # print(self.cands_dataframe)
        self.__normalize_experience_scores()
        # print(self.cands_dataframe)
        self.__create_final_scores_dataframe()
        # print(self.cands_dataframe)
        self.__calc_project_count_final_normalized_scores()
        # print(project_experiences)
        self.__add_cand_score_reasons()
        # final_scores = self.__calc_final_scores()
        pd.DataFrame.to_excel(self.cands_final_score_dataframe, f"./results/jdk_{self.jdk_id}.xlsx")
        pd.DataFrame.to_excel(self.cands_dataframe, f"./results/jdk_{self.jdk_id}_all.xlsx")

        # final_project_exp_scores_dict = self.__combine_project_exp_scores(cand_ids, num_projects, project_names, project_scores_list, project_experiences)
        # final_scores_dict = self.__get_candidates_final_scores_dict(final_project_exp_scores_dict, skill_scores_dict)
        
        # return final_scores_dict
        return 1


    def score_candidates(self):
        self.jdk_embedding = self.__fetch_jdk_embeddings()
        
        candidate_scores = self.__get_all_candidate_scores()
        
        # save_to_excel(self.jdk_id, candidate_scores)


def score_candidates(jdk_id, candidate_id_list):
    _ = load_dotenv(find_dotenv())

    # print(os.environ.get('PINECONE_API_KEY'))
    pinecone.init(      
        api_key=os.environ.get('PINECONE_API_KEY'),
        environment='gcp-starter'
    )

    jdk_resume_assistant = HRAssistant(jdk_id, candidate_id_list)
    jdk_resume_assistant.score_candidates()
