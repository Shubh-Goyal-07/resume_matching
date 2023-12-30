from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv, find_dotenv

import openai
import os

import numpy as np

import pinecone

import time
import math

from save_scores import save_to_excel



class JDKResumeAssistant():
    def __init__(self, jdk_id, candidate_id_list):
        self.jdk_id = jdk_id
        self.candidate_id_list = candidate_id_list
        
        self.index = pinecone.Index("willings")


    def __fetch_jdk_embeddings(self):
        jdk_id_str = str(self.jdk_id)

        jdk_embedding = self.index.fetch(ids=[jdk_id_str], namespace="jdks").to_dict()
        jdk_embedding = jdk_embedding['vectors'][jdk_id_str]['values']

        jdk_skill_embeddings = self.index.fetch(ids=[jdk_id_str], namespace="jdks_skills").to_dict()
        jdk_skill_embeddings = jdk_skill_embeddings['vectors'][jdk_id_str]['values']

        return jdk_embedding, jdk_skill_embeddings


    def __fetch_candidate_scores(self, candidate_id):
        candidate_namespace = f"candidate_{candidate_id}"
        candidate_skill_namespace = f"candidate_{candidate_id}_skills"

        cand_project_query_response = self.index.query(
                                            vector=self.jdk_embedding,
                                            namespace=candidate_namespace,
                                            top_k=10,
                                            include_values=False
                                        )['matches']
        
        cand_skill_query_response = self.index.query(
                                            vector=self.jdk_skill_embeddings,
                                            namespace=candidate_skill_namespace,
                                            top_k=10,
                                            include_values=False
                                        )['matches']
        
        
        project_scores = {}
        skill_scores = {}

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


        for cand_skill in cand_skill_query_response:
            cand_skill_id = cand_skill['id']
            cand_skill_score = cand_skill['score']

            skill_scores[cand_skill_id] = round(cand_skill_score, 2)

        return project_scores, skill_scores


    def __get_normalized_project_scores(self, project_scores_list):
        
        mean = max(np.float64(0.75), np.mean(project_scores_list))
        project_devs_factor = project_scores_list - mean
        project_devs_factor = list(map(lambda x: max(round(2-math.exp(-3*x), 2), 0), project_devs_factor))

        project_scores_normal = list(np.around(100 * np.array(project_scores_list) * np.array(project_devs_factor), 2))
        
        return project_scores_normal


    def __get_normalized_experience_scores(self, project_experiences):
        percentile_75 = np.percentile(project_experiences, 75)
        percentile_50 = np.percentile(project_experiences, 50)
        percentile_25 = np.percentile(project_experiences, 25)

        # print(percentile_75, percentile_50, percentile_25)
        print(project_experiences)
        project_experiences_normalised = list(map(lambda x: 1 if x>=percentile_75 else (0.9 if x>=percentile_50 else (0.8 if x>=percentile_25 else 0.7)), project_experiences))

        return project_experiences_normalised


    def __get_candidates_final_scores_dict(self, project_scores_dict, skill_scores_dict):
        final_scores = []

        for candidate_id in project_scores_dict:
            final_score = 0

            cand_project_scores = project_scores_dict[candidate_id]
            
            for project_skill in cand_project_scores:
                score_pro = cand_project_scores[project_skill]
                score_pro = min(score_pro, 100)
                # score_pro = 0 if score_pro<60 else score_pro
                cand_project_scores[project_skill] = score_pro

            valid_num = 0
            for project_skill in cand_project_scores:
                final_score += cand_project_scores[project_skill]
                valid_num += 1 if cand_project_scores[project_skill] else 0

            final_score = final_score/valid_num if valid_num else 0

            final_scores.append({"id": candidate_id, "final_score": final_score, "project_scores": [{"name": project, "score": cand_project_scores[project]} for project in cand_project_scores]})

        return final_scores


    def __extract_data_lists(self, project_scores):
        project_names = []
        project_scores_list = []
        num_projects = []
        cand_ids = []
        project_experiences = []

        for candidate_id in project_scores:
            cand_ids.append(candidate_id)
            
            projects = project_scores[candidate_id]

            num_projects.append(len(projects))
            
            for project in projects:
                project_scores_list.append(projects[project])

                name, experience = project.split("__")
                project_names.append(name)
                project_experiences.append(float(experience))

        return cand_ids, num_projects, project_names, project_scores_list, project_experiences


    def __combine_project_exp_scores(self, cand_ids, num_projects, project_names, project_scores_list, project_experiences):
        # project_exp_scores = list(np.around(np.array(project_scores_list) * np.array(project_experiences), 2))
        project_exp_scores = project_scores_list
        num_cand = len(cand_ids)

        final_project_exp_scores_dict = {}
        for i in range(num_cand):
            final_project_exp_scores_dict[cand_ids[i]] = {}
            for j in range(num_projects[i]):
                final_project_exp_scores_dict[cand_ids[i]][project_names.pop(0)] = project_exp_scores.pop(0)

        return  final_project_exp_scores_dict 


    def __get_all_candidate_scores(self):
        
        project_scores_all = {}
        skill_scores_dict = {}

        for candidate_id in self.candidate_id_list:
            project_scores, skill_scores = self.__fetch_candidate_scores(candidate_id)
            project_scores_all[candidate_id] = project_scores
            skill_scores_dict[candidate_id] = skill_scores

        cand_ids, num_projects, project_names, project_scores_list, project_experiences = self.__extract_data_lists(project_scores_all)
        
        project_scores_list = self.__get_normalized_project_scores(project_scores_list)
        project_experiences = self.__get_normalized_experience_scores(project_experiences)
        
        # print(project_experiences)

        final_project_exp_scores_dict = self.__combine_project_exp_scores(cand_ids, num_projects, project_names, project_scores_list, project_experiences)
        final_scores_dict = self.__get_candidates_final_scores_dict(final_project_exp_scores_dict, skill_scores_dict)
        
        return final_scores_dict


    def score_candidates(self):
        self.jdk_embedding, self.jdk_skill_embeddings = self.__fetch_jdk_embeddings()
        
        candidate_scores = self.__get_all_candidate_scores()
        
        save_to_excel(self.jdk_id, candidate_scores)



def get_candidate_score(jdk_id, candidate_id_list):
    _ = load_dotenv(find_dotenv())

    # print(os.environ.get('PINECONE_API_KEY'))
    pinecone.init(      
        api_key=os.environ.get('PINECONE_API_KEY'),
        environment='gcp-starter'
    )

    jdk_resume_assistant = JDKResumeAssistant(jdk_id, candidate_id_list)
    jdk_resume_assistant.score_candidates()


# if __name__ == "__main__":
#     get_candidate_score(3, [1, 2, 3])