from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv, find_dotenv

import openai
import os

import numpy as np

import pinecone

import math

import pandas as pd

import json


class HRAssistant():
    def __init__(self, jdk_info, candidates_info):
        self.jdk_id = jdk_info['id']
        self.jdk_desc = jdk_info['description']

        self.candidate_id_list = []
        self.candidate_desc_list = []

        for candidate in candidates_info:
            self.candidate_id_list.append(candidate['id'])
            self.candidate_desc_list.append(candidate['description'])
        
        self.index = pinecone.Index("willings")


        config = json.load(open('./config.json'))
        self.sim_score_penalty_params = config['similarity_score_penalty_params']
        self.experience_penalties = config['experience_params']['experience_percentile_penalties']
        self.project_count_penalties = config['project_count_penalties']


    def __fetch_jdk_embeddings(self):
        jdk_id_str = str(self.jdk_id)

        jdk_embedding = self.index.fetch(ids=[jdk_id_str], namespace="jdks").to_dict()
        jdk_embedding = jdk_embedding['vectors'][jdk_id_str]['values']

        return jdk_embedding


    def __fetch_candidate_scores(self, candidate_id):
        candidate_namespace = f"candidate_{candidate_id}"

        cand_project_query_response = self.index.query(
                                            vector=self.jdk_embedding,
                                            namespace=candidate_namespace,
                                            top_k=10,
                                            include_values=False
                                        )['matches']
        
        
        project_scores = {}

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
        minimum_mean = self.sim_score_penalty_params['minimum_mean']
        dev_e_factor = self.sim_score_penalty_params['dev_e_factor']
        cutoff = self.sim_score_penalty_params['cutoff_score_after_penalty']
        
        # Step 1: Get the mean of the column
        project_score_mean = max(self.cands_dataframe['project_score'].mean(), minimum_mean)

        # Step 2: Subtract the mean from the column to create a new column
        self.cands_dataframe['project_devs'] = self.cands_dataframe['project_score'] - project_score_mean

        # Step 3: Apply a lambda function on the new column
        self.cands_dataframe['project_devs'] = self.cands_dataframe['project_devs'].apply(lambda x: max(round(2-math.exp(-dev_e_factor*x), 2), 0))

        # Step 4: Multiply the first column and the new column and store the value in the first column
        self.cands_dataframe['project_score'] = self.cands_dataframe['project_score'] * self.cands_dataframe['project_devs']

        # Step 5: Delete the new column
        self.cands_dataframe.drop('project_devs', axis=1, inplace=True)

        # Step 6: Apply a lambda function to the first column
        self.cands_dataframe['project_score'] = self.cands_dataframe['project_score'].apply(lambda x: 0 if x<cutoff else round(x*100, 2))

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

        penalty_75_100 = self.experience_penalties['75_100']
        penalty_50_75 = self.experience_penalties['50_75']
        penalty_25_50 = self.experience_penalties['25_50']
        penalty_0_25 = self.experience_penalties['0_25']


        self.cands_dataframe['experience'] = self.cands_dataframe['experience'].apply(lambda x: penalty_75_100 if x>=percentile_75th else (penalty_50_75 if x>=percentile_50th else (penalty_25_50 if x>=percentile_25th else penalty_0_25)))

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
                penalties = self.project_count_penalties['mode_1']['more_than_50']
            else:
                penalties = self.project_count_penalties['mode_1']['less_than_50']
        
        elif mode==2:
            percentage_345 = sum(percentage_list[2:]) 
            if percentage_345 > self.project_count_penalties['equivalence_factor']*percentage_list[1]:
                penalties = self.project_count_penalties['mode_2']['3_5_equivalent']
            else:
                penalties = self.project_count_penalties['mode_2']['normal']
        
        elif mode==3:
            percentage_45 = sum(percentage_list[3:])
            if percentage_45 > self.project_count_penalties['equivalence_factor']*percentage_list[2]:
                penalties = self.project_count_penalties['mode_3']['4_5_equivalent']
            else:
                penalties = self.project_count_penalties['mode_3']['normal']

        elif mode==4:
            penalties = self.project_count_penalties['mode_4']
        
        elif mode==5:
            penalties = self.project_count_penalties['mode_5']

        
        self.cands_final_score_dataframe['project_count'] = self.cands_final_score_dataframe['project_count'].apply(lambda x: penalties[x-1])
        self.cands_final_score_dataframe['final_score'] = self.cands_final_score_dataframe['final_score'] * self.cands_final_score_dataframe['project_count']
        self.cands_final_score_dataframe['final_score'] = self.cands_final_score_dataframe['final_score'].apply(lambda x: round(x, 2))
        
        self.cands_final_score_dataframe.drop('project_count', axis=1, inplace=True)


        return


    def __create_llm_chain(self):
        template = """You are a reasoning agent. We have job description for a job position in the field of technology.
        Multiple candidates applied for the job. All of them submitted their resumes and we have calculated a score that shows the aptness of the applicant for the job position.
        
        We will give you a job description and the set of projects of the applicant alongwith the score that we calculated. You have to analyse the job description, the projects, and provide a reasoning for why the applicant has been given that score.

        The score is given out of 100. A candidate may get a high, low, or a moderate score. So carefully analyze the job description, the projects and then provide a reasoning as to why the applicant has a particular score. Say an applicant has a bad score then you need to justify how the applicant is not so well suited for the job based on the job description and the applicant's projects. Similarly if the applicant has a high score then you need to provide a reasoning as to why the applicant is suited for the job.
        

        {jdk_description}

        The candidate has worked on the following projects: {candidate_description}.

        The candidate has been given a score of {candidate_score}.

        You have to return the output in the following format. Remember to be very brief while providing the reasoning. Try not to exceed 60 words.
        
        Reasoning: <A VERY SUCCINT REASONING>"""

        prompt = PromptTemplate(template=template, input_variables=["jdk_description", "candidate_description", "candidate_score"])

        self.llm_chain = LLMChain(prompt=prompt, llm=OpenAI())

        return


    def __add_cand_score_reasons(self):
        self.__create_llm_chain()

        for index, row in self.cands_final_score_dataframe.iterrows():
            candidate_id = row['id']
            candidate_score = row['final_score']
            candidate_projects_info = self.candidate_desc_list[self.candidate_id_list.index(candidate_id)]

            final_reason = self.llm_chain.run(jdk_description=self.jdk_desc, candidate_description=candidate_projects_info, candidate_score=candidate_score)
            final_reason = final_reason.split("Reasoning: ")[-1]

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

        self.cands_dataframe = self.__create_dataframe(project_scores_all)
        self.__normalize_project_scores()
        self.__drop_irrelevant_projects()
        self.__normalize_experience_scores()
        self.__create_final_scores_dataframe()
        self.__calc_project_count_final_normalized_scores()
        self.__add_cand_score_reasons()
        # pd.DataFrame.to_excel(self.cands_final_score_dataframe, f"./results/jdk_{self.jdk_id}.xlsx", index=False)
        # pd.DataFrame.to_excel(self.cands_dataframe, f"./results/jdk_{self.jdk_id}_all.xlsx", index=False)

        result_data_json = self.cands_final_score_dataframe.to_json(orient='records')        
        return result_data_json


    def score_candidates(self):
        self.jdk_embedding = self.__fetch_jdk_embeddings()
        candidate_scores = self.__get_all_candidate_scores()
        return candidate_scores


def score_candidates(jdk_info, candidates_info):
    _ = load_dotenv(find_dotenv())

    # print(os.environ.get('PINECONE_API_KEY'))
    pinecone.init(      
        api_key=os.environ.get('PINECONE_API_KEY'),
        environment='gcp-starter'
    )

    jdk_resume_assistant = HRAssistant(jdk_info, candidates_info)
    result = jdk_resume_assistant.score_candidates()

    return result
