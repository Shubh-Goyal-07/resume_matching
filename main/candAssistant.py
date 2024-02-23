import openai
from pinecone import Pinecone

from dotenv import load_dotenv, find_dotenv
import os

import numpy as np
import pandas as pd
import math

import json


class JobSearchAssistant():
    """
    This class contains the methods to score the suggested jobs based on the candidate's projects and skills.

    The flow of the scoring process is as follows:
    1. Fetch the candidate's embedding from the pinecone index.
    2. Fetch the scores of the top 50 suggested jobs from the pinecone index.
    3. Create a dataframe containing the job ids and scores for the suggested jobs.
    4. Normalize the scores of the suggested jobs.
    5. Integrate the skill based scoring with the job scores.
    6. Add the reasoning for the score of the job.
    
    Attributes:
    ----------
    __candidate_id : str
        The id of the candidate.

    __candidate_description : str
        The description of the candidate which was generated during upserting the candidate's resume into the pinecone index.

    __candidate_skills : list
        The list of skills of the candidate.

    __jdk_id_list : list
        The list of job ids.

    __jdk_description_list : list
        The list of job descriptions.

    __client : openai.OpenAI
        The instance of the OpenAI class.

    __pinecone_config : dict
        The dictionary containing the configuration for the pinecone index.

    __index : pinecone.Index
        The index object for the pinecone index of the index name "willings".

    __dev_e_factor : float
        The penalty factor for normalizing the scores of the suggested jobs.

    __skill_count_percentile_penalties : dict
        The dictionary containing the penalty factors for normalizing the scores of the suggested jobs based on the skills of the candidate.

    jdk_dataframe : pandas.DataFrame
        The dataframe containg the scores of the suggested jobs.

    Methods:
    -------
    __init__(self, candidate_info, jdks_info)
        The constructor for the JobSearchAssistant class.

    __fetch_candidate_embedding(self)
        Fetches the embedding of the corresponding candidate's description from the pinecone index.

    __fetch_jdk_scores(self)
        Fetches the scores of the jobs based on the candidate's description.

    __create_jdk_dataframe(self, jdk_query_scores)
        Creates a dataframe containing the job ids and scores for the suggested jobs.

    __normalize_jdk_scores(self)
        Normalizes the scores of the suggested jobs based on the deviation of the scores from the mean score.

    __integrate_score_skills(self)
        Assigns a skill score for the suggested jobs based on the skills required for the job and the skills of the candidate.

    __get_score_reasoning(self, jdk_description, jdk_score)
        Generates a reasoning for the score of the job description with respect to the candidate's resume.

    __add_job_score_reasons(self)
        Adds the reasoning for the score of the job description with respect to the candidate's resume to the dataframe.

    suggest_jobs(self)
        Scores the suggested jobs based on the candidate's resume.
    """

    def __init__(self, candidate_info, jdks_info):
        """
        The constructor for the JobSearchAssistant class.

        Parameters:
        ----------
        candidate_info : dict
            The dictionary containing the candidate's id and resume.
            The dictionary should contain the following
            - id (str) : The id of the candidate.
            - description (str) : The description of the candidate which was generated during upserting the candidate's resume into the pinecone index.
        """

        # Load .env file
        _ = load_dotenv(find_dotenv())

        # Extracting the candidate's id, description and skills from the candidate_info dictionary
        self.__candidate_id = candidate_info['id']
        self.__candidate_description, self.__candidate_skills = candidate_info['description'].split("SKILLS: ")
        self.__candidate_skills = self.__candidate_skills.split(", ")

        # Extracting the job ids and descriptions from the jdks_info dictionary
        self.__jdk_id_list = []
        self.__jdk_description_list = []

        for jdk in jdks_info:
            self.__jdk_id_list.append(str(jdk['id']))
            self.__jdk_description_list.append(jdk['description'])

        # Set the openai api key and create an instance of the OpenAI class
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        self.__client = openai.OpenAI()

        # Load the configuration file
        config = json.load(open('./config.json'))
        # Load the pinecone config
        self.__pinecone_config = config['pinecone_config']

        # Create an instance of the Pinecone class
        pinecone_key = os.environ.get('PINECONE_API_KEY')
        pc = Pinecone(
            api_key=pinecone_key,
            environment='gcp-starter'
        )

        # Create an index object for the pinecone index of the index name "willings"
        self.__index = pc.Index(self.__pinecone_config['index_name'])

        # Penalty factors for normalizing the scores of the suggested jobs
        self.__dev_e_factor = config['job_suggestion_dev_e_factor']
        self.__skill_count_percentile_penalties = config['candAsst_skill_count_penalties']

    def __fetch_candidate_embedding(self):
        """
        Fetches the embedding of the corresponding candidate's description from the pinecone index.

        Parameters:
        ----------
        None

        Returns:
        -------
        cand_embedding : list
            The embedding of the corresponding candidate's resume.
        """

        # Convert the candidate id to a string
        # As the fetch method of the pinecone index requires the ids to be strings
        cand_id_str = str(self.__candidate_id)

        # Fetch the candidate's embedding from the pinecone index
        # Creating list as pinecone fetch method requires ids to be in list
        ids = [cand_id_str]
        namespace = self.__pinecone_config['candidate_description_namespace']
        cand_embedding = self.__index.fetch(
            ids=ids, namespace=namespace).to_dict()
        cand_embedding = cand_embedding['vectors'][cand_id_str]['values']

        return cand_embedding

    def __fetch_jdk_scores(self):
        """
        Fetches the scores of the jobs based on the candidate's description.

        Parameters:
        ----------
        None

        Returns:
        -------
        jdk_query_response : list
            The list of dictionaries containing the job ids and scores for the suggested jobs. The list contains the job ids and scores for the suggested jobs.
        """

        namespace = self.__pinecone_config['jdk_namespace']

        # Query the pinecone index to get the scores of the suggested jobs
        # The top_k parameter is set to 50 to get the top 50 suggested jobs
        jdk_query_response = self.__index.query(
            vector=self.candidate_embedding,
            namespace=namespace,
            top_k=50,
            include_values=False,
        )['matches']

        return jdk_query_response

    def __create_jdk_dataframe(self, jdk_query_scores):
        """
        Creates a dataframe containing the job ids and scores for the suggested jobs.

        Parameters:
        ----------
        jdk_query_scores : list
            The list of dictionaries containing the job ids and scores for the suggested jobs.
            The dicttionary contains the following:
            - id (str) : The id of the job.
            - score (float) : The score of the job.

        Returns:
        -------
        jdk_dataframe : pandas.DataFrame
            A dataframe containing the following columns:
                - id : The id of the job.
                - score : The score of the job.
                - description : The description of the job.
                - skills : The skills required for the job.
            Each row corresponds to one job.
        """

        # Lists to store the data of the jobs
        jdk_ids = []
        jdk_scores = []
        jdk_descs = []
        jdk_skills = []

        # Traverse through the jdk_query_scores list and extract the job ids and scores
        for jdk in jdk_query_scores:
            # Check if the job id is in the list of job ids required for the candidate
            if jdk['id'] not in self.__jdk_id_list:
                continue

            jdk_ids.append(jdk['id'])
            jdk_scores.append(jdk['score'])

            # Extract the description and skills of the job from the jdk_description_list using the job id
            # The index of the job id in the jdk_id_list is used to get the corresponding description and skills
            desc, skills = self.__jdk_description_list[self.__jdk_id_list.index(jdk['id'])].split("SKILLS: ")
            skills = skills.split(", ")                     # Split the skills into a list

            jdk_descs.append(desc)
            jdk_skills.append(skills)

        # Create a dictionary that can be directly converted to a pandas dataframe
        jdk_data_dict = {"id": jdk_ids, "score": jdk_scores, "description": jdk_descs, "skills": jdk_skills}
        # Create a dataframe from the dictionary
        jdk_dataframe = pd.DataFrame(jdk_data_dict)

        return jdk_dataframe

    def __normalize_jdk_scores(self):
        """
        This method normalizes the scores of the suggested jobs based on the deviation of the scores from the mean score.

        The method applies the following steps:
        1. Calculate the mean score of the suggested jobs.
        2. Calculate the deviation of the scores from the mean score.
        3. Multiply the scores with the following exponential function:
            (2 - e^(-deviation)) OR 0 if the result is negative
            Here, dev_e_factor is an arbitrary factor that is used to control the steepness of the exponential function.

            (This is done on the same column as step 2 to avoid creating a new column.)

        4. Convert the scores to integers and cap the scores to 100.

        This normalization penalizes the job scores more than it rewards them for the same amount of deviation from the mean score.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

        # Calculate the mean score of the suggested jobs
        jdk_score_mean = self.jdk_dataframe['score'].mean()

        # Calculate the deviation of the scores from the mean score
        self.jdk_dataframe['score_devs'] = self.jdk_dataframe['score'] - \
            jdk_score_mean
        
        # Apply exponential function to the deviation of the scores from the mean score and get the penalty values
        self.jdk_dataframe['score_devs'] = self.jdk_dataframe['score_devs'].apply(
            lambda x: max(round(2-math.exp(-self.__dev_e_factor*x), 2), 0))
        
        # Multiply the scores with the penalty values
        self.jdk_dataframe['score'] = self.jdk_dataframe['score'] * \
            self.jdk_dataframe['score_devs']
        self.jdk_dataframe.drop('score_devs', axis=1, inplace=True)

        # Convert the scores to integers and cap the scores to 100
        self.jdk_dataframe['score'] = self.jdk_dataframe['score'].apply(
            lambda x: min(int(x*100), 100))
        self.jdk_dataframe.sort_values(
            by=['score'], ascending=False, inplace=True)

        return

    def __integrate_score_skills(self):
        """
        This method assigns a skill score for the suggested jobs based on the skills required for the job and the skills of the candidate.
        The score is then multiplied with the job score to get the final score for the job.

        The method applies the following steps:
        1. Find the number of common skills between the candidate and the job.
        2. Calculate the percentile breakpoints for the skills column.
        3. Assign the penalty factors to the skills column based on the percentiles.
        4. Multiply the job scores with the skill penalty factors.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        
        """

        # Find the number of common skills between the candidate and the job
        self.jdk_dataframe['skills'] = self.jdk_dataframe['skills'].apply(lambda x: len(list(set(x).intersection(self.__candidate_skills))))

        # Calculate the percentile breakpoints for the skills column
        column_percentiles = np.percentile(
            self.jdk_dataframe['skills'], [90, 75, 50, 25])

        percentile_90th = column_percentiles[0]
        percentile_75th = column_percentiles[1]
        percentile_50th = column_percentiles[2]
        percentile_25th = column_percentiles[3]

        # Get the penalty factors for each percentile slots from the config file
        # Slots being 0-25, 25-50, 50-75, 75-90, 90-100
        penalty_90_100 = self.__skill_count_percentile_penalties['90_100']
        penalty_75_90 = self.__skill_count_percentile_penalties['75_90']
        penalty_50_75 = self.__skill_count_percentile_penalties['50_75']
        penalty_25_50 = self.__skill_count_percentile_penalties['25_50']
        penalty_0_25 = self.__skill_count_percentile_penalties['0_25']

        # Assign the penalty factors to the skills column based on the percentiles
        self.jdk_dataframe['skills'] = self.jdk_dataframe['skills'].apply(lambda x: penalty_90_100 if x >= percentile_90th else (
            penalty_75_90 if x >= percentile_75th else (
                penalty_50_75 if x >= percentile_50th else (
                    penalty_25_50 if x >= percentile_25th else penalty_0_25
                )
            )
        ))

        # Multiply the job scores with the skill penalty factors
        self.jdk_dataframe['score'] = self.jdk_dataframe['score'] * self.jdk_dataframe['skills']

        # Drop the skills column
        self.jdk_dataframe.drop('skills', axis=1, inplace=True)

        return

    def __get_score_reasoning(self, jdk_description, jdk_score):
        """
        This method generates a reasoning for the score of the job description with respect to the candidate's resume.

        Uses the OpenAI GPT-3.5-turbo model to generate the reasoning.

        Parameters:
        ----------
        jdk_description : str
            The description of the job which was returned during the upserting of the job description into the pinecone index.
        jdk_score : int
            The score of the job description with respect to the candidate's resume.

        Returns:
        -------
        reasoning : str
            The reasoning for the score of the job description with respect to the candidate's resume.
        """

        # Defines the system and user prompts for the OpenAI GPT-3.5-turbo model
        system_prompt = """You are a reasoning agent who is trying to help a candidate get a job and reasons out why a particular job is suitable or unsuitable for the candidate."""

        user_prompt = f"""We have the resume of the candidate. We had a list of job descriptions that might or might not be suitable for the candidate. So we have calculated a score for each job description with respect to the candidate's resume. The score is out of 100. The higher the score the more suitable the job is for the candidate.

        We will give you a job description and the set of projects of the candidate alongwith the score that was obtained for that particular job description. You have to analyse the job description, the projects, and provide a reasoning for why the job is suitable or not suitable for the candidate.

        A job may get a high, low, or a moderate score. So carefully analyze the job description, the projects and then provide a reasoning as to why the job has a particular relevance score with respect to the candidate's resume. Say a job description is given a low score then you need to provide a reasoning as to why that job is not suitable for the candidate. Similarly, if a job description is given a high score then you need to provide a reasoning as to why that job is suitable for the candidate. If a job description is given a moderate score then you need to provide a reasoning as to why that job is neither suitable nor unsuitable for the candidate.

        Make sure to keep a note of the following points while reasoning:

        1. Give the reasoning without mentioning about the candidate's skills and experience in detail.

        2. Focus more on how the job is suitable or unsuitable for the candidate and less on how the candidate is suitable or unsuitable for the job.

        3. Give the reasoning in second person pronouns, that is, as if you are telling the candidate why the job is suitable or unsuitable for them.

        The candidate has worked on the following projects: {self.__candidate_description}.

        The job description is as follows: {jdk_description}.

        The job has been given a score of {jdk_score}.

        You have to return the output in the following format. Remember to be very brief while providing the reasoning. Try not to exceed 60 words.

        Reasoning: <A VERY SUCCINT REASONING>"""

        # Generate a response for the system and user prompts using the model
        model = "gpt-3.5-turbo"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        reasoning = self.__client.chat.completions.create(model=model, messages=messages)

        # Extract the reasoning from the response
        reasoning = reasoning.choices[0].message.content
        reasoning = reasoning.split("Reasoning: ")[-1]

        return reasoning

    def __add_job_score_reasons(self):
        """
        This method adds the reasoning for the score of the job description with respect to the candidate's resume to the dataframe.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

        # Loop through the dataframe and add the reasoning for the score of the job description with respect to the candidate's resume
        for index, row in self.jdk_dataframe.iterrows():
            # Get the job id, score and description
            jdk_id = row['id']
            jdk_score = row['score']
            jdk_description = row['description']

            # Get the reasoning for the score of the job description with respect to the candidate's resume
            reasoning = self.__get_score_reasoning(jdk_description, jdk_score)

            # Add the reasoning to the dataframe in the 'reason' column
            self.jdk_dataframe.at[index, 'reason'] = reasoning

        # Drop the description column
        self.jdk_dataframe.drop('description', axis=1, inplace=True)

        return

    def suggest_jobs(self):
        """
        This function can be called to get the job suggestions for the candidate.

        It applies the following steps:
        1. Fetch the candidate's embedding from the pinecone index.
        2. Fetch the scores of the top 50 suggested jobs from the pinecone index.
        3. Create a dataframe containing the job ids and scores for the suggested jobs.
        4. Normalize the scores of the suggested jobs.
        5. Integrate the skill based scoring with the job scores.
        6. Add the reasoning for the score of the job.

        Parameters:
        ----------
        None

        Returns:
        -------
        str
            The 'jdk_dataframe' dataframe in form of a JSON string.
            Each dictionary in the JSON string contains the following:
            - id : The id of the job.
            - score : The score of the job.
            - reason : The reasoning for the score of the job.

        Optional output:
        -------
        Uncomment the relevant line in the method to save the dataframe to an excel file.
        """

        # Step 1: Fetch the candidate's embedding from the pinecone index
        self.candidate_embedding = self.__fetch_candidate_embedding()
        
        # Step 2: Fetch the scores of the top 50 suggested jobs from the pinecone index
        jdk_query_scores = self.__fetch_jdk_scores()
        
        # Step 3: Create a dataframe containing the job ids and scores for the suggested jobs
        self.jdk_dataframe = self.__create_jdk_dataframe(jdk_query_scores)

        # Step 4: Normalize the scores of the suggested jobs
        self.__normalize_jdk_scores()

        # Step 5: Integrate the skill based scoring with the job scores
        self.__integrate_score_skills()

        # step 6: Add the reasoning for the score of the joB
        self.__add_job_score_reasons()

        # Convert the dataframe to a JSON object
        result_data_json = self.jdk_dataframe.to_json(orient='records')
        
        # Uncomment the below line to save the 'jdk_dataframe' dataframe to an excel file
        # pd.DataFrame.to_excel(self.jdk_dataframe, f"./job_suggestions/candidate_{self.__candidate_id}.xlsx", index=False)

        return result_data_json


def get_job_suggestions(candidate_info, jdks_info):
    """
    This function can be called to get the job suggestions for the candidate.

    Parameters:
    ----------
    candidate_info : dict
        The dictionary containing the candidate's id and resume.
        The dictionary should contain the following
        - id (str) : The id of the candidate.
        - description (str) : The description of the candidate which was generated during upserting the candidate's resume into the pinecone index.
    
    jdks_info : list
        The list of dictionaries containing the job ids and descriptions.
        The dictionary should contain the following
        - id (str) : The id of the job.
        - description (str) : The description of the job.

    Returns:
    -------
    results : str
        The 'jdk_dataframe' dataframe in form of a JSON string.
        Each dictionary in the JSON string contains the following:
        - id : The id of the job.
        - score : The score of the job.
        - reason : The reasoning for the score of the job.
    """

    cand_assistant = JobSearchAssistant(candidate_info, jdks_info)
    results = cand_assistant.suggest_jobs()
    return results