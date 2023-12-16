import os
from openai import OpenAI

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
        embeddings = self.client.embeddings.create(input = [skill], model="text-embedding-ada-002").data[0].embedding
        return embeddings


    def __calculate_similarity(self, skill1, skill2):
        tensor1 = self.__get_skill_embeddings(skill1)
        tensor2 = self.__get_skill_embeddings(skill2)
        cos_sim = dot(tensor1, tensor2)/(norm(tensor1)*norm(tensor2))
        return cos_sim
    

    def __get_project_skill_scores(self, project_skills):
        project_skill_scores = []

        for project_skill in project_skills:
            max_val = 0
            for skill in self.jdk_skills:
                similarity = self.__calculate_similarity(skill, project_skill)
                if similarity > max_val:
                    max_val = similarity
            project_skill_scores.append(max_val)
            
        return project_skill_scores
    

    def get_candidate_score(self):
        self.__load_openai_client()

        score = 0
        for project in self.projects:
            project_skill_scores = self.__get_project_skill_scores(self.projects[project]['skills'])
            project_relevance_score = self.projects[project]['relevance_score']
            score +=  project_relevance_score*(sum(project_skill_scores)/len(project_skill_scores))*self.projects[project]['experience']
            print(f"project: {project} Done")

        score = score/len(self.projects)
        return score



def match_skills(jdk_skills, candidate_project_dic):

    resume_scores = []
    for candidate in candidate_project_dic:
        jdk_candidate_match = JDK_skills(jdk_skills, candidate_project_dic[candidate])
        resume_score = jdk_candidate_match.get_candidate_score()
        resume_scores.append({'id': candidate, 'score': resume_score})
        print(f"candidate: {candidate} Done")
    
    return resume_scores