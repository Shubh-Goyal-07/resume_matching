from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv, find_dotenv

import openai
import os

import pinecone

import time

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
    

    def __get_project_scores_from_query_scores(self, cand_project_query_response, cand_skill_query_response):
        scores = {}

        for cand_project in cand_project_query_response:
            cand_project_id = cand_project['id']
            cand_project_score = cand_project['score']

            scores[cand_project_id] = 10*cand_project_score


        for cand_skill in cand_skill_query_response:
            cand_skill_id = cand_skill['id']
            cand_skill_score = cand_skill['score']

            if cand_skill_id in scores:
                scores[cand_skill_id] *= 10*cand_skill_score

        # print(scores)
        project_scores = {}
        for project in scores:
            project_name, project_exp = project.split("__")
            project_scores[project_name] = scores[project] * float(project_exp)

        # print(project_scores)
        return project_scores
    

    def __get_candidate_score(self, candidate_id):
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
        

        project_scores = self.__get_project_scores_from_query_scores(cand_project_query_response, cand_skill_query_response)
        project_scores = dict(sorted(project_scores.items(), key=lambda x: x[1], reverse=True))

        # print(project_scores)

        final_score = 0
        for project in project_scores:
            final_score += project_scores[project]

        final_score = final_score/len(project_scores)

        return final_score, project_scores
    

    def __get_all_candidate_scores(self):
        candidate_scores = []

        for candidate_id in self.candidate_id_list:
            # start_time = time.time()
            final_score, project_scores = self.__get_candidate_score(candidate_id)
            candidate_scores.append({
                "id": candidate_id,
                "final_score": final_score,
                "project_scores": [{"name": project, "score": project_scores[project]} for project in project_scores]
            })
            # print("--- %s seconds ---" % (time.time() - start_time))

        return candidate_scores
    

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