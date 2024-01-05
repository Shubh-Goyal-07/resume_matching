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

import pandas as pd

import json

# from googletrans import Translator


class JobSearchAssistant():
    def __init__(self, candidate_id):
        self.candidate_id = candidate_id
        
        self.index = pinecone.Index("willings")


        config = json.load(open('./config.json'))
        self.dev_e_factor = config['job_suggestion_dev_e_factor']


    def __fetch_candidate_embedding(self):
        cand_id_str = str(self.candidate_id)

        cand_embedding = self.index.fetch(ids=[cand_id_str], namespace="all_candidates").to_dict()
        cand_embedding = cand_embedding['vectors'][cand_id_str]['values']

        return cand_embedding


    def __fetch_jdk_scores(self):
        namespace = f"jdks"

        jdk_query_response = self.index.query(
                                            vector=self.candidate_embedding,
                                            namespace=namespace,
                                            top_k=50,
                                            include_values=False
                                        )['matches']
        
        return jdk_query_response
    

    def __create_jdk_dataframe(self, jdk_query_scores):
        jdk_ids = []
        jdk_scores = []

        for jdk in jdk_query_scores:
            jdk_ids.append(jdk['id'])
            jdk_scores.append(jdk['score'])

        jdk_data_dict = {"id": jdk_ids, "score": jdk_scores}
        jdk_dataframe = pd.DataFrame(jdk_data_dict)

        return jdk_dataframe


    def __normalize_jdk_scores(self):
        jdk_score_mean = self.jdk_dataframe['score'].mean()
        self.jdk_dataframe['score_devs'] = self.jdk_dataframe['score'] - jdk_score_mean
        self.jdk_dataframe['score_devs'] = self.jdk_dataframe['score_devs'].apply(lambda x: min(max(round(2-math.exp(-self.dev_e_factor*x), 2), 0), 100))
        self.jdk_dataframe['score'] = self.jdk_dataframe['score'] * self.jdk_dataframe['score_devs']
        self.jdk_dataframe.drop('score_devs', axis=1, inplace=True)
        self.jdk_dataframe['score'] = self.jdk_dataframe['score'].apply(lambda x: int(x*100))
        self.jdk_dataframe.sort_values(by=['score'], ascending=False, inplace=True)

        return
    

    def suggest_jobs(self):
        self.candidate_embedding = self.__fetch_candidate_embedding()
        jdk_query_scores = self.__fetch_jdk_scores()
        self.jdk_dataframe = self.__create_jdk_dataframe(jdk_query_scores)
        self.__normalize_jdk_scores()
        result_data_json = self.jdk_dataframe.to_json(orient='records')
        
        return result_data_json
    

def get_job_suggestions(candidate_id):
    _ = load_dotenv(find_dotenv())

    # print(os.environ.get('PINECONE_API_KEY'))
    pinecone.init(      
        api_key=os.environ.get('PINECONE_API_KEY'),
        environment='gcp-starter'
    )

    jdk_resume_assistant = JobSearchAssistant(candidate_id)
    results = jdk_resume_assistant.suggest_jobs()
    return results
