import openai
from pinecone import Pinecone
from google.cloud import translate_v2 as translate

from dotenv import load_dotenv, find_dotenv
import os

import pandas as pd
import numpy as np
import math

import json
import time

import tiktoken


class HRAssistant():
    """
    This class contains the methods to score the candidates based on their projects for a particular job description.

    The flow of the scoring process is as follows:
    1. Fetch the job description and the candidate's projects from the pinecone index.
    2. Normalize the project scores of the candidates through an exponential function.
    3. Drop the projects which have a score less than the cutoff score.
    4. Assign a experience score to each project based on the experiences of the candidates on each project.
    5. Averages the projects of each individual candidate and creates a new dataframe containing the final scores of the candidates.
    6. Calculates the final scores of the candidates scores by penalizing the scores based on the project count of the candidates.
    7. Generates the reasoning for a candidate's score and also generates the personality score of the candidate along with the reasoning.
    8. Adds the reasoning for the candidate scores and the personality scores of the candidates to the dataframe 'cands_final_score_dataframe'.

    Attributes:
    ----------
    __jdk_id : str
        The id of the jdk.

    __jdk_desc : str
        The description of the jdk.

    __jdk_tech_skills : str
        The technical skills required for the jdk.

    __jdk_soft_skills : str
        The soft skills required for the jdk.

    __candidate_id_list : list
        The list of the ids of the candidates.

    __candidate_desc_list : list
        The list of the descriptions of the candidates.

    __candidate_recruit_answers : list  
        The list of the answers of the candidates for the screening questions and the questionnaire.

    __client : openai.OpenAI
        The instance of the OpenAI class.

    __pinecone_config : dict
        The dictionary containing the configuration of the pinecone index.

    __index : pinecone.Index
        The index object for the pinecone index of the index name "willings".

    __sim_score_penalty_params : dict
        The dictionary containing the parameters for the similarity score penalty.

    __experience_penalties : dict
        The dictionary containing the parameters for the experience penalties.

    __project_count_penalties : dict
        The dictionary containing the parameters for the project count penalties.

    __cands_dataframe : pandas.DataFrame
        A dataframe containing the project scores of the candidates.

    cands_final_score_dataframe : pandas.DataFrame
        A dataframe containing the final scores of the candidates.

    Methods:
    --------
    __init__(jdk_info, candidates_info)
        The constructor for HRAssistant class.

    __fetch_jdk_embeddings()
        Fetches the embedding of the jdk from the pinecone index.

    __fetch_candidate_scores(candidate_id)
        Fetches the scores of the projects of the candidate.

    __create_dataframe(project_scores)
        Creates a dataframe containing the project scores of the candidates.

    __normalize_project_scores()
        Normalizes the project scores of the candidates through an exponential function.

    __drop_irrelevant_projects()
        Drops the projects which have a score less than the cutoff score.

    __normalize_experience_scores()
        Assigns a experience score to each project based on the experiences of the candidates on each project.

    __create_final_scores_dataframe()
        Averages the projects of each individual candidate in the dataframe 'cands_dataframe' and creates a new dataframe containing the final scores of the candidates.

    __calc_project_count_final_normalized_scores()
        Calculates the final scores of the candidates scores by penalizing the scores based on the project count of the candidates.

    __get_score_reasons_and_personality_scores(candidate_recruit_answers, candidate_score, candidate_description)
        Generates the reasoning for a candidate's score and also generates the personality score of the candidate along with the reasoning.

    __translate_en_ja(reason)
        Translates a text (in any language) to Japanese using the Google Cloud Translate API.

    __add_reasons_and_scores()
        Adds the reasoning for the candidate scores and the personality scores of the candidates to the dataframe 'cands_final_score_dataframe'.

    score_candidates()
        Scores the candidates based on their projects for a particular job description.
    """

    def __init__(self, jdk_info, candidates_info, cand_count_reasoning):
        """
        The constructor for HRAssistant class.

        Parameters:
        ----------
        jdk_info : dict
            The dictionary containing the data of the jdk.
            The dictionary should contain the following
            - id (str) : The id of the jdk.
            - description (str) : The description of the jdk which was generated during upserting the jdk.

        candidates_info : list
            The list of dictionaries containing the data of the candidates.
            Each dictionary should contain the following
            - id (str) : The id of the candidate.
            - description (str) : The description of the candidate which was generated during upserting the candidate.
            - galkRecruitScreeningAnswers (dict) : The dictionary containing the answers of the candidate for the screening questions.
            - galkRecruitQuestionnaireAnswers (dict) : The dictionary containing the answers of the candidate for the questionnaire.

        Returns:
        -------
        None
        """

        # Load .env file
        _ = load_dotenv(find_dotenv())

        # Extracting the jdk data from the jdk_info dictionary
        self.__jdk_id = jdk_info['id']
        self.__jdk_desc = jdk_info['description']
        self.__jdk_desc, self.__jdk_tech_skills = self.__jdk_desc.split(
            "SKILLS: ")
        self.__jdk_soft_skills = jdk_info['softSkills']

        # Extracting the candidate data from the candidates_info list
        self.__candidate_id_list = []
        self.__candidate_desc_list = []
        self.__candidate_recruit_answers = []
        self.__candidate_name = []
        self.__candidate_img = []
        self.__candidate_college = []
        self.__candidate_major = []

        for candidate in candidates_info:
            self.__candidate_id_list.append(candidate['id'])
            self.__candidate_desc_list.append(candidate['description'])

            self.__candidate_name.append(candidate['name'])
            self.__candidate_img.append(candidate['img'])
            self.__candidate_college.append(candidate['collegeName'])
            self.__candidate_major.append(candidate['major'])

            # Creating one dictionary of the answers of the candidate for the screening questions and the questionnaire
            answers = candidate['galkRecruitScreeningAnswers']
            answers.update(candidate['galkRecruitQuestionnaireAnswers'])
            self.__candidate_recruit_answers.append(str(answers))

        self.__cand_raw_data_dataframe = pd.DataFrame(
            {"id": self.__candidate_id_list, "description": self.__candidate_desc_list, "answers": self.__candidate_recruit_answers, "name": self.__candidate_name, "img": self.__candidate_img, "collegeName": self.__candidate_college, "major": self.__candidate_major})

        # delete the temporary lists
        del answers, self.__candidate_desc_list, self.__candidate_recruit_answers, self.__candidate_name, self.__candidate_img, self.__candidate_college, self.__candidate_major

        # Defines the number of candidates to produce reasoning for
        self.cand_count_reasoning = cand_count_reasoning

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

        # Similarity score penalty parameters (applied on each project)
        self.__sim_score_penalty_params = config['similarity_score_penalty_params']
        # Experience months percentile penalties (applied on each project)
        self.__experience_penalties = config['experience_params']['experience_percentile_penalties']
        # Project count penalties (applied on the final score of the candidate)
        self.__project_count_penalties = config['project_count_penalties']

    def __fetch_jdk_embeddings(self):
        """
        Fetches the embedding of the jdk from the pinecone index through job id

        Parameters:
        ----------
        None

        Returns:
        -------
        jdk_embedding : list
            The embedding of the job description.
        """

        # Convert the jdk_id to a string (in case it is not a string)
        # As the vector ids of upserted vectors are always strings
        jdk_id_str = str(self.__jdk_id)

        # Fetch the embedding of the jdk from the pinecone index
        # creating list as pinecone fetch function takes list of ids
        ids = [jdk_id_str]
        namespace = self.__pinecone_config['jdk_namespace']
        jdk_embedding = self.__index.fetch(
            ids=ids, namespace=namespace).to_dict()
        jdk_embedding = jdk_embedding['vectors'][jdk_id_str]['values']

        return jdk_embedding

    def __fetch_candidate_scores(self, candidate_id):
        """
        Fetches the scores of the projects of the candidate.

        Parameters:
        ----------
        candidate_id : str
            The id of the candidate whose projects are to be fetched from the pinecone index.

        Returns:
        -------
        project_scores : dict
            The dictionary containing the scores of the projects of the candidate.
        """

        projects_namespace = self.__pinecone_config['projects_namespace']

        # Create filter to fetch the project scores of a particular candidate (searches in metadata)
        filter = {
            "candidate_id": {"$eq": f'{candidate_id}'},
        }
        # Fetch the projects of the candidate from the pinecone index
        # The query will return the top 10 projects of the candidate
        # Returns cosine similarity scores of the projects with the jdk_embedding
        cand_project_query_response = self.__index.query(
            vector=self.__jdk_embedding,
            namespace=projects_namespace,
            filter=filter,
            top_k=10,
            include_values=False,
            include_metadata=True
        )['matches']

        # Dictionary to store the scores of the projects of the candidate
        project_scores = {}

        # Iterate through the project in the query response and store the scores in the dictionary
        for cand_project in cand_project_query_response:
            # experience stored in metadata
            experience = cand_project['metadata']['experience']
            cand_project_id = f"{cand_project['id']}__{experience}"
            cand_project_score = cand_project['score']

            # Apply some enhancements and penalties to the scores
            # Reason: 0.9 is a very high score for cosine and is rarely reached
            #         <0.75 is a very low score for cosine
            if cand_project_score >= 0.9:
                cand_project_score = 1
            elif cand_project_score >= 0.8:
                cand_project_score += 0.05
            elif cand_project_score <= 0.75:
                cand_project_score -= 0.05

            # Round off the score to 2 decimal places
            project_scores[cand_project_id] = round(cand_project_score, 2)

        return project_scores

    def __create_dataframe(self, project_scores):
        """
        Creates a dataframe containing the project scores of the candidates.

        Parameters:
        ----------
        project_scores : dict
            The dictionary containing the scores of the projects of the candidate.
            The dictionary contains the following:
            - The key is the id of the candidate.
            - The value is a dictionary containing following:
                - The key is the id of the project.
                - The value is the score of the project.

        Returns:
        -------
        cands_dataframe : pandas.DataFrame
            A dataframe containing the following columns:
                - id : The id of the candidate.
                - name : The name of the project.
                - project_score : The score of the project.
                - experience : The experience of the candidate.
            Each row corresponds to only one candidate-project pair.
        """

        # Lists to store the data of the candidates
        project_names = []
        project_scores_list = []
        cand_ids = []
        project_experiences = []

        # Traverse through each candidate
        for candidate_id in project_scores:
            projects = project_scores[candidate_id]
            # Traverse through each project of the candidate
            for project in projects:
                cand_ids.append(candidate_id)
                project_scores_list.append(projects[project])

                # Split the project id to get the name and the experience
                # Example: "1__Project1__2" -> "1", "Project1", "2"
                _, name, experience = project.split("__")
                project_names.append(name)
                project_experiences.append(float(experience))

        # Create a dictionary that can be directly converted to a dataframe
        cands_data_dict = {"id": cand_ids, "name": project_names,
                           "project_score": project_scores_list, "experience": project_experiences}

        # Create a dataframe from the dictionary
        cands_dataframe = pd.DataFrame(cands_data_dict)

        return cands_dataframe

    def __normalize_project_scores(self):
        """
        This function normalizes the project scores of the candidates through an exponential function.

        The function applies the following steps:
        1. Get the mean of the project scores
        2. Subtract the mean from the project scores (i.e. get the deviation) to create a new column
        3. Multiply the project scores with the following exponential function:
                            2 - exp(-deviation*dev_e_factor) OR 0 if the result is negative
                    Here, dev_e_factor is a an arbitrary constant taken from config file

            (This is done on the same column as step 2 to avoid creating a new column and save memory)

        4. Multiply the project scores with 100 and rounf off to 2 decimal places

        This normalization penalizes the projects more than it rewards for the same amount of deviation from the mean.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

        # Minimum mean parameter defines the lower bound of the project score means
        # This ensures that if all the projects are irrelevant, then rather than rewarding any, it penalizes all of them
        minimum_mean = self.__sim_score_penalty_params['minimum_mean']
        dev_e_factor = self.__sim_score_penalty_params['dev_e_factor']

        # Step 1: Get the mean of the project scores
        project_score_mean = max(
            self.__cands_dataframe['project_score'].mean(), minimum_mean)

        # Step 2: Subtract the mean from the column to get the deviation from the score mean and store the deviation onto column 'project_devs'
        self.__cands_dataframe['project_devs'] = self.__cands_dataframe['project_score'] - \
            project_score_mean

        # Step 3: Apply exponential function on the deviation column amd get the penalty values
        self.__cands_dataframe['project_devs'] = self.__cands_dataframe['project_devs'].apply(
            lambda x: max(round(2-math.exp(-dev_e_factor*x), 2), 0))

        # Step 4: Multiply the project scores and the penalty values
        self.__cands_dataframe['project_score'] = self.__cands_dataframe['project_score'] * \
            self.__cands_dataframe['project_devs']

        # Step 5: Delete the project_devs column
        self.__cands_dataframe.drop('project_devs', axis=1, inplace=True)

        # Step 6: Multiply the final project scores by 100 and round off to 2 decimal places
        self.__cands_dataframe['project_score'] = self.__cands_dataframe['project_score'].apply(
            lambda x: round(x*100, 2))

        return

    def __drop_irrelevant_projects(self):
        """
        Drops the projects which have a score less than the cutoff score.

        This step ensures that the projects with less scores are not considered for the final score. Otherwise, they may affect the final score negatively.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

        # Get the cutoff score
        cutoff = self.__sim_score_penalty_params['cutoff_score_after_penalty']*100

        # Drop the projects with scores less than the cutoff score and make a new dataframe
        temp_dataframe = self.__cands_dataframe[self.__cands_dataframe['project_score'] >= cutoff]

        # If all the projects had score less than cutoff then do not drop any project
        if len(temp_dataframe) != 0:
            self.__cands_dataframe = temp_dataframe

        return

    def __normalize_experience_scores(self):
        """
        This function assigns a experience score to each project based on the experiences of the candidates on each project.
        The score of the project is then multiplied by the experience score.

        The experience score is calculated as follows:
        1. Get the 75th, 50th, and 25th percentiles of the experience column
        2. Assign experience scores to the experiences based on the percentiles
        3. Multiply the project scores with the experience scores

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

        # Calculate the percentile breakpoints of the experience column
        column_percentiles = np.percentile(
            self.__cands_dataframe['experience'], [75, 50, 25])

        percentile_75th = column_percentiles[0]
        percentile_50th = column_percentiles[1]
        percentile_25th = column_percentiles[2]

        # Get the experience scores for each percentile slot from the config file
        # Slots being 75-100, 50-75, 25-50, 0-25
        penalty_75_100 = self.__experience_penalties['75_100']
        penalty_50_75 = self.__experience_penalties['50_75']
        penalty_25_50 = self.__experience_penalties['25_50']
        penalty_0_25 = self.__experience_penalties['0_25']

        # Assign the experience scores to the experiences based on the percentiles
        self.__cands_dataframe['experience'] = self.__cands_dataframe['experience'].apply(lambda x: penalty_75_100 if x >= percentile_75th else (
            penalty_50_75 if x >= percentile_50th else (penalty_25_50 if x >= percentile_25th else penalty_0_25)))

        # Multiply the project scores with the experience scores
        self.__cands_dataframe['final_score'] = self.__cands_dataframe['project_score'] * \
            self.__cands_dataframe['experience']

        # Round off the final scores to 2 decimal places and limit the maximum score to 100
        self.__cands_dataframe['final_score'] = self.__cands_dataframe['final_score'].apply(
            lambda x: min(round(x, 2), 100))

        return

    def __create_final_scores_dataframe(self):
        """
        This function averages the projects of each individual candidate in the dataframe 'cands_dataframe' and creates a new dataframe containing the final scores of the candidates.

        It also calculates the project count of each candidate and stores it in the new dataframe 'cands_final_score_dataframe'.    

        The 'cands_final_score_dataframe' contains the following columns:
        - id : The id of the candidate.
        - final_score : The final score of the candidate.
        - project_count : The project count of the candidate.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

        # Calculate the project count of each candidate
        self.__cands_dataframe['project_count'] = self.__cands_dataframe.groupby(
            'id')['final_score'].transform('count')

        # Calculate the final scores of the candidates
        # Sum over the final scores of the projects of each candidate
        self.cands_final_score_dataframe = self.__cands_dataframe.groupby('id').agg(
            {'final_score': 'sum', 'project_count': 'first'}).reset_index()

        # Divide the sum by the project count to get the average
        self.cands_final_score_dataframe['final_score'] = self.cands_final_score_dataframe['final_score'] / \
            self.cands_final_score_dataframe['project_count']

        return

    def __calc_project_count_final_normalized_scores(self):
        """
        This function calculates the final scores of the candidates scores by penalizing the scores based on the project count of the candidates.

        The following steps are performed:
        1. The project count of each candidate is capped at 5.
        2. The mode of the project count is calculated.
        3. The percentage of the mode is calculated.
        4. The penalties are calculated based on the mode and the percentage.
        5. The final scores are multiplied by the penalties.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

        # Step 1: Cap the project count at 5
        total_entries = len(self.cands_final_score_dataframe)
        self.cands_final_score_dataframe['project_count'] = self.cands_final_score_dataframe['project_count'].apply(
            lambda x: min(x, 5))

        # Step 2: Calculate the mode of the project count
        # Create a dataframe containing the count of each possible project count i.e. 1, 2, 3, 4, 5
        count_dataframe = pd.DataFrame(
            self.cands_final_score_dataframe['project_count'].value_counts())
        count_dataframe['percentage'] = (
            count_dataframe['count'] / total_entries) * 100

        # Step 3: Calculate the percentage of the mode
        # List to store the percentages of each project count (0th index for project count 1)
        percentage_list = [0, 0, 0, 0, 0]
        max_percentage = 0
        for index, row in count_dataframe.iterrows():
            percentage_list[index-1] = row['percentage']
            if row['percentage'] >= max_percentage:
                max_percentage = row['percentage']
                mode = index

        # Step 4: Calculate the penalties based on the mode and the percentage
        # The penalties are stored in the config file
        penalties = [1, 1, 1, 1, 1]
        if mode == 1:
            if percentage_list[0] >= 50:
                penalties = self.__project_count_penalties['mode_1']['more_than_50']
            else:
                penalties = self.__project_count_penalties['mode_1']['less_than_50']

        elif mode == 2:
            percentage_345 = sum(percentage_list[2:])
            if percentage_345 > self.__project_count_penalties['equivalence_factor']*percentage_list[1]:
                penalties = self.__project_count_penalties['mode_2']['3_5_equivalent']
            else:
                penalties = self.__project_count_penalties['mode_2']['normal']

        elif mode == 3:
            percentage_45 = sum(percentage_list[3:])
            if percentage_45 > self.__project_count_penalties['equivalence_factor']*percentage_list[2]:
                penalties = self.__project_count_penalties['mode_3']['4_5_equivalent']
            else:
                penalties = self.__project_count_penalties['mode_3']['normal']

        elif mode == 4:
            penalties = self.__project_count_penalties['mode_4']

        elif mode == 5:
            penalties = self.__project_count_penalties['mode_5']

        # Step 5: Multiply the final scores by the penalties
        self.cands_final_score_dataframe['project_count'] = self.cands_final_score_dataframe['project_count'].apply(
            lambda x: penalties[x-1])                       # x-1 because the list is 0-indexed
        self.cands_final_score_dataframe['final_score'] = self.cands_final_score_dataframe['final_score'] * \
            self.cands_final_score_dataframe['project_count']

        # Round off the final scores to 2 decimal places
        self.cands_final_score_dataframe['final_score'] = self.cands_final_score_dataframe['final_score'].apply(
            lambda x: round(x, 2))
        
        # Sort the dataframe by the final scores in descending order
        self.cands_final_score_dataframe = self.cands_final_score_dataframe.sort_values(by='final_score', ascending=False)

        # Drop the project count column to save memory
        self.cands_final_score_dataframe.drop(
            'project_count', axis=1, inplace=True)

        return

    def __get_score_reasons_and_personality_scores(self, candidate_recruit_answers, candidate_score, candidate_description):
        """
        This function generates the reasoning for a candidate's score on the basis of his/her projects and skills and the job description and skills required by the company.

        Additionally, it also generates the personality score (out of 5) of the candidate along with the reasoning based on the answers given by the candidate on a fixed set of questions and the soft skills required by the company.

        Uses the OpenAI GPT-3.5-turbo-1106 model to generate the reasoning and the personality score.
        The model is used because it supports the chat completions and the JSON response format.

        Parameters:
        ----------
        candidate_recruit_answers : str
            The answers given by the candidate for the screening questions and the questionnaire.

        candidate_score : float
            The score of the candidate.

        candidate_description : str
            The description of the candidate which was generated during upserting the candidate.

        Returns:
        -------
        tuple[str, float, str]
            The reasoning for the candidate's score, the personality score of the candidate, and the reasoning for the personality score.
        """

        # Get the candidate's projects and technical skills
        candidate_description, candidate_tech_skills = candidate_description.split(
            "SKILLS: ")

        # Defines the system and user prompts for the OpenAI model
        system_prompt = """You are a hiring recruiter's assisstant who is finding best candidates for recruitment and reasons out why a particular candidate is suitable or unsuitable for the job based on the candidate's past projects and skills. Aditionally, you also help the recruiter judge an applicant's personality and his/her willingness to go to Japan to work for the company. You have to provide the recruiter raw reviews of the candidates based on your assessment of the candidate's projects and skills and his/her personality and willingness to work in Japan. The recruiter will then use your reviews to make the final decision."""

        user_prompt = f"""We have job description for a job position in the field of technology.
        Multiple candidates applied for the job. All of them submitted their resumes and we have calculated a score that shows the aptness of the applicant for the job position. You are supposed to carry out the following 2 tasks.

        TASK-1: Providing a reasoning for the suitability/unsuitability of the applicant for the job based on his/her projects and skills

        We will give you a job description and the set of projects of the applicant alongwith the score that we calculated. You have to analyse the job description, the projects, and provide a reasoning for why the applicant has been given that score.

        The score is given out of 100. An applicant may get a high, low, or a moderate score. So carefully analyze the job description, the projects and then provide a reasoning as to why the applicant has a particular score. Say an applicant has a bad score then you need to justify how the applicant is not so well suited for the job based on the job description and the applicant's projects. Similarly if the applicant has a high score then you need to provide a reasoning as to why the applicant is suited for the job.

        Furthermore, the resume also contains the skills of the applicant. Alongwith the job description, the company has also provided us with a list of skills that it requires the applicants to have. You will be given a list of both the skills, the applicant skills as well as the skills required by the company. Your task is to first list out the skills that are common in both, and then the skills that are required but are not possessed by the applicant. You have to provide a reasoning based on those skills that are required but are not possessed by the applicant.


        You need to consider the following points while providing the reasoning:

        1. Do not use any modal verbs that will indicate a probability of any kind.

        2. Do not mention anything about the score that has been given to the candidate. You have to provide a reasoning based on the job description and the projects of the applicant and not on the score of the candidate. Use the score for your internal understanding only.

        3. Consider both the job description and the projects of the applicant while providing the reasoning. Additionally, consider the skills that are required by the company and the skills that the applicant possesses. You have to provide a reasoning based on the skills as well. If some skills are required by the company but are not possessed by the applicant then you have to provide a reasoning based on those skills as well. Try to beautify the positive points.

        4. Keep the reasoning more towards the technical side and try to keep it as positive as possible.

        5. Also, make sure not to mention anything about soft skills in the technical reasoning as it has to be strictly based on the projects and the skills of the applicant which match or differ from the skills required by the company.
        And strictly remember to mention the similar skills and the skills that are required but are not possessed by the applicant in the technical reasoning as it is very important as per the company's policy.

        6. Do not show the irrelevant projects of the applicant in a positive light. If the project is irrelevant then avoid mentioning the project in the reasoning. If you use the project in the reasoning then it should be used to show the applicant's skills in a positive light but not relevancy of the project.

        7. If very less skills are common between the applicant and the company then critcize the candidate a bit and do not strongly support the candidate.

        8. Strictly use simple level english vocabulary and grammar which can be easily understood by the recruiter and can be easily translated to japanese. The is no need to use any complex words or sentences.

        9. Write the reasoning such that it has been written by a human and not by a machine.


        The job description provided by the company: {self.__jdk_desc}

        The applicant has worked on the following projects: {candidate_description}.

        The applicant has been given a score of {candidate_score}.

        Required skills set shared by the company: {self.__jdk_tech_skills}.
        Applicant skills: {candidate_tech_skills}.


        TASK-2: Determining the applicant's personality and his/her willingness to work in Japan based on the answers given by him/her

        As mentioned earlier, one of your jobs is to determine the applicant's personality and his/her willingness to work in Japan. To do this task, you will be given a JSON containing all the questions which were asked to the candidate along with the answers that the applicant gave. You have to give a single score based on the applicant's personality and his/her willingness to go to Japan.

        While judging the personality of the applicant, you also have to consider the soft skills that the company is looking for in a candidate. Be sure to consider those skills as they are very important for the company.

        The soft skills that the company is looking for are: {self.__jdk_soft_skills}.

        The question answers given by the applicant: {candidate_recruit_answers}.

        You need to keep the following points in mind while providing the score:

        1. Make sure not to mention any technical reasoning, skills or any other stuff that is not related to the personality of the applicant. Do not mention 'AI/ML' or 'Python' or 'Java' or any other technical stuff.

        2. Do not mention anything about the score that has been given to the candidate. You have to provide a reasoning based on the company's soft skill requirement and the answers given by the candidate and not on the score of the candidate. Use the score for your internal understanding only.

        3. There should be no line in the reasoning mentioning the candidate got this much score out of 5.

        Make sure not to mention score in any of the technical of soft skill score reasonings in any way. It is strictly prohibited to mention the score in the reasoning as per the company's policy.

        
        Additionaly, you have to give the reasons in japanese language as well.
        Keep the following points in consideration while translating:

        1. Translate it to an easy to understand native style japanese language.
        
        2. Do not change the meaning of the text while translating.  Also, make sure to use the correct grammar and vocabulary.

        
        You have to return the output of all the above 2 tasks in the following JSON format:
            tech_reason: <GIVE THE REASONING HERE THAT IS ASKED FOR IN TASK-1>,
            tech_reason_japanese: <GIVE THE REASONING IN JAPANESE LANGUAGE HERE THAT IS ASKED FOR IN TASK-1>,
            score: <GIVE A SCORE OUT OF 5 HERE FOR TASK-2>,
            reason: <GIVE A REASON FOR THE SCORE YOU GAVE IN TASK-2>,
            reason_japanese: <GIVE THE REASON IN JAPANESE LANGUAGE HERE FOR THE SCORE YOU GAVE IN TASK-2>
        """


        # encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-1106")
        # enc1 = encoding.encode(system_prompt)
        # enc2 = encoding.encode(user_prompt)

        # Generate a response for the system and user prompts using the model
        # The response is in the JSON format
        model = "gpt-3.5-turbo-1106"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            # strt_time = time.time()
            response = self.__client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=messages,
            )

            response = json.loads(
                response.choices[0].message.content, strict=False)
            
            # Extract the reasoning and the personality score and it's reasoning from the response
            techreason = response['tech_reason']
            # Cap the score between 0 and 5
            score = int(round(max(min(response['score'], 5), 0), 0))
            reason = response['reason']
            techreason_jap = response['tech_reason_japanese']
            reason_jap = response['reason_japanese']

            return [techreason, techreason_jap, score, reason, reason_jap]

            # print(response.usage)
            # print("Time taken for OpenAI: ", time.time() - strt_time)
            # Convert the response to a dictionary
            
        except openai.RateLimitError:
            # print("OpenAI limit reahced.")
            time.sleep(10)
            
            return self.__get_score_reasons_and_personality_scores(candidate_recruit_answers, candidate_score, candidate_description)

    def __add_reasons_and_scores(self):
        """
        This function adds the reasoning for the candidate scores and the personality scores of the candidates to the dataframe 'cands_final_score_dataframe'.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

        running_count = 0

        # Iterate through the dataframe and add the reasoning and the personality scores to the dataframe
        for index, row in self.cands_final_score_dataframe.iterrows():
            # Get the candidate's id and score in the current row
            candidate_id = row['id']
            candidate_score = row['final_score']

            # Extract the candidate's information from the dataframe '__cand_raw_data_dataframe'
            # cand_info will be a pd.Series object
            cand_info = self.__cand_raw_data_dataframe[self.__cand_raw_data_dataframe['id'] == candidate_id].iloc[0]

            # Get the recruit answers and the project descriptions from the row of the candidate
            candidate_recruit_answers = cand_info.at['answers']
            candidate_projects_info = cand_info.at['description']

            # check if the specified number of candidates' reasons has been added (self.cand_count_reasoning)
            running_count += 1
            if (running_count < self.cand_count_reasoning):
                # Get the reasoning for the candidate's score and the personality score of the candidate
                response = self.__get_score_reasons_and_personality_scores(
                    candidate_recruit_answers, candidate_score, candidate_projects_info)
                
                techreason, techreason_jap, score, personalityreason, personalityreason_jap = response
            else:
                techreason = techreason_jap = score = personalityreason = personalityreason_jap = None
                
            # Add the technical reasoning and the personality scores and reasoning to the dataframe 'cands_final_score_dataframe'
            self.cands_final_score_dataframe.loc[index,
                                                 'tech_reason'] = techreason
            self.cands_final_score_dataframe.loc[index,
                                                 'tech_reason_ja'] = techreason_jap
            self.cands_final_score_dataframe.loc[index,
                                                 'personality_score'] = score
            self.cands_final_score_dataframe.loc[index,
                                                 'personality_reason'] = personalityreason
            self.cands_final_score_dataframe.loc[index,
                                                 'personality_reason_ja'] = personalityreason_jap
            
            # add value from name, img, collegeName and major
            self.cands_final_score_dataframe.loc[index, 'name'] = cand_info.at['name']
            self.cands_final_score_dataframe.loc[index, 'img'] = cand_info.at['img']
            self.cands_final_score_dataframe.loc[index, 'collegeName'] = cand_info.at['collegeName']
            self.cands_final_score_dataframe.loc[index, 'major'] = cand_info.at['major']
            
        return

    def score_candidates(self):
        """
        This function can be called to score the candidates based on their projects for a particular job description.

        It uses the given jdk and the candidates' projects to calculate the final scores of the candidates in the following steps:
        1. Fetch the embedding of the jdk from the pinecone index.
        2. Loop through the candidate ids and fetch the scores of the projects of the candidates.
        3. Create a dataframe containing the project scores of the candidates, 'cands_dataframe'.
        4. Normalize the project scores of the candidates and drop the irrelevant projects and apply the experience scores normalization.
        5. Create a dataframe containing the final scores of the candidates, 'cands_final_score_dataframe'.
        6. Calculate the final scores of the candidates and the project count penalties.
        7. Add the reasoning for the candidate scores and the personality scores of the candidates to the dataframe 'cands_final_score_dataframe'.

        Parameters:
        ----------
        None

        Returns:
        -------
        str
            The 'cands_final_score_dataframe' in form of a JSON object.
            Each dictionary in the list contains the following:
            - id (str): The unique id of the candidate.
            - final_score (float): The final score of the candidate for the jdk.
            - tech_reason (str): The reasoning behind the final score.
            - tech_reason_japanese (str): tech_reason translated to Japanese.
            - personality_score (int): Soft skills based personality score of the candidate.
            - personality_reason (str): The reasoning behind the personality_score.
            - personality_reason_japanese (str): personality_reason translated to Japanese.

        Optional output:
        ---------------
        Uncomment the relevant lines to save the intermediate DataFrames (`cands_dataframe` and `cands_final_score_dataframe`) as Excel files in the './results' folder.
        """

        # Step 1: Fetch the embedding of the jdk from the pinecone index
        self.__jdk_embedding = self.__fetch_jdk_embeddings()

        # Step 2: Loop through the candidate ids and fetch the scores of the projects of the candidates
        project_scores_all = {}

        for candidate_id in self.__candidate_id_list:
            project_scores = self.__fetch_candidate_scores(candidate_id)
            project_scores_all[candidate_id] = project_scores

        # Step 3: Create a dataframe containing the project scores of the candidates, 'cands_dataframe'
        self.__cands_dataframe = self.__create_dataframe(project_scores_all)

        # Step 4: Normalize the project scores of the candidates and drop the irrelevant projects and apply the experience scores normalization
        self.__normalize_project_scores()
        self.__drop_irrelevant_projects()
        self.__normalize_experience_scores()

        # Step 5: Create a dataframe containing the final scores of the candidates, 'cands_final_score_dataframe'
        self.__create_final_scores_dataframe()

        # Step 6: Calculate the final scores of the candidates and the project count penalties
        self.__calc_project_count_final_normalized_scores()

        # Step 7: Add the reasoning for the candidate scores and the personality scores of the candidates to the dataframe 'cands_final_score_dataframe'
        self.__add_reasons_and_scores()

        # Convert the dataframe to a JSON object
        result_data_json = self.cands_final_score_dataframe.to_json(
            orient='records')
        
        # Uncomment the below line to save the 'cands_datframe', which contains the project scores
        # pd.DataFrame.to_excel(self.__cands_dataframe, f"./results/{self.__jdk_id}.xlsx", index=False)

        # Uncomment the below line to save the 'cands_final_score_dataframe', which contains the final scores of the candidates (FINAL RESULT)
        pd.DataFrame.to_excel(self.cands_final_score_dataframe,
                              f"./results/jdk1_{self.__jdk_id}.xlsx", index=False)

        return result_data_json


def get_candidate_scores(jdk_info, candidates_info, cand_count=30):
    """
    This function scores the candidates based on their projects for a particular job description.

    Parameters:
    ----------
    jdk_info : dict
        The dictionary containing the data of the jdk.
        The dictionary should contain the following
        - id (str) : The id of the jdk.
        - description (str) : The description of the jdk which was generated during upserting the jdk.

    candidates_info : list
        The list of dictionaries containing the data of the candidates.
        Each dictionary should contain the following
        - id (str) : The id of the candidate.
        - description (str) : The description of the candidate which was generated during upserting the candidate.
        - galkRecruitScreeningAnswers (dict) : The dictionary containing the answers of the candidate for the screening questions.
        - galkRecruitQuestionnaireAnswers (dict) : The dictionary containing the answers of the candidate for the questionnaire.

    Returns:
    -------
    str
        The 'cands_final_score_dataframe' in form of a JSON object.
        Each dictionary in the list contains the following:
        - id (str): The unique id of the candidate.
        - final_score (float): The final score of the candidate for the jdk.
        - tech_reason (str): The reasoning behind the final score.
        - tech_reason_japanese (str): tech_reason translated to Japanese.
        - personality_score (int): Soft skills based personality score of the candidate.
        - personality_reason (str): The reasoning behind the personality_score.
        - personality_reason_japanese (str): personality_reason translated to Japanese.
    """

    jdk_resume_assistant = HRAssistant(jdk_info, candidates_info, cand_count)
    result = jdk_resume_assistant.score_candidates()

    return result
