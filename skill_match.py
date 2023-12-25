import os
from openai import OpenAI
import time

import torch.nn.functional as F
from numpy import dot
from numpy.linalg import norm


class JDK_skills():
    def __init__(self, jdk_skills, projects):
        self.jdk_skills = jdk_skills
        self.projects = projects


    def __load_openai_client(self):
        self.client = OpenAI()
        self.client.api_key = os.getenv("OPENAI_API_KEY")

        return
            

    def __get_skill_embeddings(self, skill):
        # print("skill: ", skill)
        embeddings = self.client.embeddings.create(input = skill, model="text-embedding-ada-002").data[0].embedding
        return embeddings


    def __calculate_similarity(self, job_skills, applicant_project_skills):
        time_start = time.time()
        tensor1 = self.__get_skill_embeddings(job_skills)
        tensor2 = self.__get_skill_embeddings(applicant_project_skills)
        cos_sim = dot(tensor1, tensor2)/(norm(tensor1)*norm(tensor2))
        # print(f"Time taken in one skill match: {time.time()-time_start}")
        return cos_sim
    

    def __get_project_skill_scores(self, project_skills):
        project_skill_scores = []

        # for project_skill in project_skills:
        #     max_val = 0
        #     for skill in self.jdk_skills:
        #         similarity = self.__calculate_similarity(skill, project_skill)
        #         if similarity > max_val:
        #             max_val = similarity
        #     project_skill_scores.append(max_val)
            
        return self.__calculate_similarity(self.jdk_skills, project_skills)
    

    def get_candidate_score(self):
        self.__load_openai_client()

        score = 0
        project_scores = []
        # print(self.projects)
        for project in self.projects:
            # print(self.projects[project])
            project_skill_score = self.__get_project_skill_scores(self.projects[project]['skills'])
            project_relevance_score = self.projects[project]['relevance_score']
            # project_experience = self.projects[project]['experience']/12
            
            # project_score = project_relevance_score*(sum(project_skill_scores)/len(project_skill_scores))*project_experience
            # project_score = project_relevance_score*(sum(project_skill_scores)/len(project_skill_scores))
            project_score = project_relevance_score*project_skill_score
            score += project_score

            project_scores.append({'name': project, 'score': project_score})

            # print(f"project: {project} Done")

        # pro = 0
        # for pro1 in self.projects:
        #     if self.projects[pro1]['relevance_score']!=0:
        #         pro+=1
        # score = score/pro
        score = score/len(self.projects)

        return score, project_scores



def match_skills(jdk_skills, candidate_project_dic):

    resume_scores = []
    for candidate in candidate_project_dic:
        jdk_candidate_match = JDK_skills(jdk_skills, candidate_project_dic[candidate])
        resume_score, project_scores = jdk_candidate_match.get_candidate_score()
        resume_scores.append({'id': candidate, 'final_score': resume_score, 'project_scores': project_scores})
        print(f"candidate: {candidate} Done")
    
    return resume_scores