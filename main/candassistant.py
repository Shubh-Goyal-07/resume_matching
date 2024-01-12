import pinecone

from dotenv import load_dotenv, find_dotenv
import os

import pandas as pd
import math

import json


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

    def __init__(self, candidate_id):
        """
        Parameters:
        ----------
        candidate_id : int
            The id of the candidate whose resume is to be used for suggesting jobs.
        
        Returns:
        -------
        None
        """
        
        self.candidate_id = candidate_id

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

        namespace = f"jdks"

        jdk_query_response = self.index.query(
            vector=self.candidate_embedding,
            namespace=namespace,
            top_k=50,
            include_values=False
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
        result_data_json = self.jdk_dataframe.to_json(orient='records')

        return result_data_json


def get_job_suggestions(candidate_id):
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

    jdk_resume_assistant = JobSearchAssistant(candidate_id)
    results = jdk_resume_assistant.suggest_jobs()
    return results
