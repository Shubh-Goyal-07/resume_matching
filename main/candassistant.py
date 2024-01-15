from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

import pinecone

from dotenv import load_dotenv, find_dotenv
import os

import pandas as pd
import math

import json

import openai

class JobSearchAssistant():
    """
    This class is used to suggest jobs to a candidate based on their resume.

    Attributes:
    ----------
    candidate_id : int
        The id of the candidate whose resume is to be used for suggesting jobs.
    index : pinecone.Index
        The pinecone index object used for querying.
    dev_e_factor : float
        The e-factor used for calculating the score deviation.
    candidate_embedding : list
        The embedding of the candidate's resume.
    jdk_dataframe : pandas.DataFrame
        The dataframe containing the job ids and scores for the suggested jobs.

    Methods:
    -------
    __fetch_candidate_embedding()
        Fetches the embedding of the candidate's resume.
    __fetch_jdk_scores()
        Fetches the scores of the jobs based on the candidate's resume.
    __create_jdk_dataframe(jdk_query_scores)
        Creates a dataframe containing the job ids and scores for the suggested jobs.
    __normalize_jdk_scores()
        Normalizes the scores of the suggested jobs.
    suggest_jobs()
        Suggests jobs to the candidate based on their resume.
    """

    def __init__(self, candidate_info, jdks_info):
        """
        Parameters:
        ----------
        candidate_id : int
            The id of the candidate whose resume is to be used for suggesting jobs.

        Returns:
        -------
        None
        """

        self.candidate_id = candidate_info['id']
        self.candidate_description = candidate_info['description']

        self.jdk_id_list = []
        self.jdk_description_list = []

        for jdk in jdks_info:
            self.jdk_id_list.append(str(jdk['id']))
            self.jdk_description_list.append(jdk['description'])

        self.index = pinecone.Index("willings")

        config = json.load(open('./config.json'))
        self.dev_e_factor = config['job_suggestion_dev_e_factor']

        self.pinecone_config = config['pinecone_config']

    def __fetch_candidate_embedding(self):
        """
        Fetches the embedding of the corresponding candidate's resume from the pinecone index.

        Parameters:
        ----------
        None

        Returns:
        -------
        cand_embedding : list
            The embedding of the corresponding candidate's resume.
        """

        cand_id_str = str(self.candidate_id)

        cand_embedding = self.index.fetch(
            ids=[cand_id_str], namespace=self.pinecone_config['candidate_description_namespace']).to_dict()
        cand_embedding = cand_embedding['vectors'][cand_id_str]['values']

        return cand_embedding

    def __fetch_jdk_scores(self):
        """
        Fetches the scores of the jobs based on the candidate's resume.

        Parameters:
        ----------
        None

        Returns:
        -------
        jdk_query_response : list
            The list of dictionaries containing the job ids and scores for the suggested jobs.
        """

        namespace = self.pinecone_config['jdk_namespace']

        jdk_query_response = self.index.query(
            vector=self.candidate_embedding,
            namespace=namespace,
            top_k=50,
            include_values=False,
            # filter={"jdk_id": {"$in": self.jdk_id_list}}
        )['matches']

        return jdk_query_response

    def __create_jdk_dataframe(self, jdk_query_scores):
        """
        Creates a dataframe containing the job ids and scores for the suggested jobs.

        Parameters:
        ----------
        jdk_query_scores : list
            The list of dictionaries containing the job ids and scores for the suggested jobs.

        Returns:
        -------
        jdk_dataframe : pandas.DataFrame
            The dataframe containing the job ids and scores for the suggested jobs.
        """

        jdk_ids = []
        jdk_scores = []

        for jdk in jdk_query_scores:
            jdk_ids.append(jdk['id'])
            jdk_scores.append(jdk['score'])

        jdk_data_dict = {"id": jdk_ids, "score": jdk_scores}
        jdk_dataframe = pd.DataFrame(jdk_data_dict)

        return jdk_dataframe

    def __normalize_jdk_scores(self):
        """
        Normalizes the scores of the suggested jobs.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

        jdk_score_mean = self.jdk_dataframe['score'].mean()
        self.jdk_dataframe['score_devs'] = self.jdk_dataframe['score'] - \
            jdk_score_mean
        self.jdk_dataframe['score_devs'] = self.jdk_dataframe['score_devs'].apply(
            lambda x: max(round(2-math.exp(-self.dev_e_factor*x), 2), 0))
        self.jdk_dataframe['score'] = self.jdk_dataframe['score'] * \
            self.jdk_dataframe['score_devs']
        self.jdk_dataframe.drop('score_devs', axis=1, inplace=True)
        self.jdk_dataframe['score'] = self.jdk_dataframe['score'].apply(
            lambda x: min(int(x*100), 100))
        self.jdk_dataframe.sort_values(
            by=['score'], ascending=False, inplace=True)

        return

    def __create_reasoning_llm_chain(self):
        """
        Creates the LLM chain for generating the reasoning.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

        template = """You are a reasoning agent. We have candidate who wants to apply for a job position in the field of technology.
        We have multiple job descriptions and the projects the candidate have worked on. And we have calculated a score for the jobs based on the candidate's projects to give the candidate a score for the job.

        We will give you a job description and the set of projects of the applicant alongwith the score that we calculated. You have to analyse the job description, the projects, and provide a reasoning for why the job has been given that score.

        The score is given out of 100. A job may get a high, low, or a moderate score. So carefully analyze the job description, the projects and then provide a reasoning as to why the job has a particular score. Say a jod has a bad score then you need to justify how the applicant is not so well suited for the job based on the job description and the applicant's projects. Similarly if the job has a high score then you need to provide a reasoning as to why the applicant is suited for the job.

        The candidate has worked on the following projects: {candidate_description}.

        The job description is as follows: {jdk_description}.

        The job has been given a score of {jdk_score}.

        You have to return the output in the following format. Remember to be very brief while providing the reasoning. Try not to exceed 60 words.
        
        Reasoning: <A VERY SUCCINT REASONING>"""

        prompt = PromptTemplate(template=template, input_variables=[
                                "jdk_description", "candidate_description", "jdk_score"])

        self.reasoning_llm_chain = LLMChain(prompt=prompt, llm=OpenAI(model='gpt-3.5-turbo-instruct'))

        return

    def __add_job_score_reasons(self):
        self.__create_reasoning_llm_chain()

        for index, row in self.jdk_dataframe.iterrows():
            jdk_id = row['id']
            jdk_score = row['score']
            jdk_description = self.jdk_description_list[self.jdk_id_list.index(jdk_id)]

            reasoning = self.reasoning_llm_chain.run(
                jdk_description=jdk_description,
                candidate_description=self.candidate_description,
                jdk_score=jdk_score
            )
            reasoning = reasoning.split("Reasoning: ")[-1]

            self.jdk_dataframe.at[index, 'reason'] = reasoning

        return

    def suggest_jobs(self):
        """
        Suggests jobs to the candidate based on their resume.

        Parameters:
        ----------
        None

        Returns:
        -------
        result_data_json : str
            The JSON string containing the job ids and scores for the suggested jobs.
        """

        self.candidate_embedding = self.__fetch_candidate_embedding()
        jdk_query_scores = self.__fetch_jdk_scores()
        self.jdk_dataframe = self.__create_jdk_dataframe(jdk_query_scores)
        self.__normalize_jdk_scores()
        self.__add_job_score_reasons()
        result_data_json = self.jdk_dataframe.to_json(orient='records')

        return result_data_json


def get_job_suggestions(candidate_info, jdks_info):
    """
    Gets the job suggestions for the candidate.

    Parameters:
    ----------
    candidate_id : int
        The id of the candidate whose resume is to be used for suggesting jobs.
    
    Returns:
    -------
    results : str
        The JSON string containing the job ids and scores for the suggested jobs.
    """
    
    _ = load_dotenv(find_dotenv())

    # print(os.environ.get('PINECONE_API_KEY'))
    pinecone.init(
        api_key=os.environ.get('PINECONE_API_KEY'),
        environment='gcp-starter'
    )

    jdk_resume_assistant = JobSearchAssistant(candidate_info, jdks_info)
    results = jdk_resume_assistant.suggest_jobs()
    return results
