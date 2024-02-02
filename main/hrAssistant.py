import openai
from pinecone import Pinecone

from dotenv import load_dotenv, find_dotenv
import os

import pandas as pd
import numpy as np
import math

import json

import time

class HRAssistant():
    """
    This class is used to score candidates based on their projects and the job description.

    Attributes:
    ----------
    jdk_id : int
        The id of the job description.
    jdk_desc : str
        The job description.
    candidate_id_list : list
        The list of ids of the candidates.
    candidate_desc_list : list
        The list of projects of the candidates.
    index : pinecone.Index
        The pinecone index object used for querying.
    sim_score_penalty_params : dict
        The parameters used for penalizing the similarity scores.
    experience_penalties : dict
        The parameters used for penalizing the experience scores.
    project_count_penalties : dict
        The parameters used for penalizing the project count scores.
    jdk_embedding : list
        The embedding of the job description.
    cands_dataframe : pandas.DataFrame
        The dataframe containing the project scores of the candidates.
    cands_final_score_dataframe : pandas.DataFrame
        The dataframe containing the final scores of the candidates.

    Methods:
    -------
    __fetch_jdk_embeddings()
        Fetches the embedding of the job description.
    __fetch_candidate_scores(candidate_id)
        Fetches the scores of the projects of the candidate.
    __create_dataframe(project_scores)
        Creates a dataframe containing the project scores of the candidates.
    __normalize_project_scores()
        Normalizes the project scores of the candidates.
    __drop_irrelevant_projects()
        Drops the projects with zero scores.
    __normalize_experience_scores()
        Normalizes the experience scores of the candidates.
    __create_final_scores_dataframe()
        Creates a dataframe containing the final scores of the candidates.
    __calc_project_count_final_normalized_scores()
        Calculates the final normalized scores of the candidates based on the project count.
    __create_llm_chain()
        Creates the LLM chain for generating the reasoning.
    __add_cand_score_reasons()
        Adds the reasoning for the candidate scores.
    __get_all_candidate_scores()
        Fetches the scores of the projects of all the candidates.
    score_candidates()
        Scores the candidates based on their projects and the job description.
    """

    def __init__(self, jdk_info, candidates_info):
        """
        Parameters:
        ----------
        jdk_info : dict
            The dictionary containing the id and the description of the job description.
        candidates_info : list
            The list of dictionaries containing the id and the description of the candidates.

        Returns:
        -------
        None
        """

        self.jdk_id = jdk_info['id']
        self.jdk_desc = jdk_info['description']
        self.jdk_desc, self.jdk_tech_skills = self.jdk_desc.split("SKILLS: ") 
        # self.jdk_soft_skills = ', '.join(jdk_info['softSkills'])
        self.jdk_soft_skills = jdk_info['softSkills']

        self.candidate_id_list = []
        self.candidate_desc_list = []
        self.candidate_recruit_answers = []

        for candidate in candidates_info:
            self.candidate_id_list.append(candidate['id'])
            self.candidate_desc_list.append(candidate['description'])
            answers = candidate['galkRecruitScreeningAnswers']
            answers.update(candidate['galkRecruitQuestionnaireAnswers'])
            self.candidate_recruit_answers.append(str(answers))

        # delete the variable answers
        del answers

        self.client = openai.OpenAI()

        pc = Pinecone(
            api_key=os.environ.get('PINECONE_API_KEY'),
            environment='gcp-starter'
        )

        self.index = pc.Index("willings")

        config = json.load(open('./config.json'))
        self.sim_score_penalty_params = config['similarity_score_penalty_params']
        self.experience_penalties = config['experience_params']['experience_percentile_penalties']
        self.project_count_penalties = config['project_count_penalties']

        self.pinecone_config = config['pinecone_config']

    def __fetch_jdk_embeddings(self):
        """
        Fetches the embedding of the job description from the pinecone index.

        Parameters:
        ----------
        None

        Returns:
        -------
        jdk_embedding : list
            The embedding of the job description.
        """

        jdk_id_str = str(self.jdk_id)

        jdk_embedding = self.index.fetch(
            ids=[jdk_id_str], namespace="jdks").to_dict()
        jdk_embedding = jdk_embedding['vectors'][jdk_id_str]['values']

        return jdk_embedding

    def __fetch_candidate_scores(self, candidate_id):
        """
        Fetches the scores of the projects of the candidate.

        Parameters:
        ----------
        candidate_id : int
            The id of the candidate whose projects are to be scored.

        Returns:
        -------
        project_scores : dict
            The dictionary containing the scores of the projects of the candidate.
        """

        projects_namespace = self.pinecone_config['projects_namespace']

        cand_project_query_response = self.index.query(
            vector=self.jdk_embedding,
            namespace=projects_namespace,
            filter={
                "candidate_id": {"$eq": f'{candidate_id}'},
            },
            top_k=10,
            include_values=False,
            include_metadata=True
        )['matches']

        # print(cand_project_query_response)

        project_scores = {}

        for cand_project in cand_project_query_response:
            experience = cand_project['metadata']['experience']
            cand_project_id = f"{cand_project['id']}__{experience}"
            cand_project_score = cand_project['score']

            if cand_project_score >= 0.9:
                cand_project_score = 1
            elif cand_project_score >= 0.8:
                cand_project_score += 0.05
            elif cand_project_score <= 0.75:
                cand_project_score -= 0.05
            elif cand_project_score <= 0.7:
                cand_project_score = 0

            if (cand_project_score):
                project_scores[cand_project_id] = round(cand_project_score, 2)

        return project_scores

    def __create_dataframe(self, project_scores):
        """
        Creates a dataframe containing the project scores of the candidates.

        Parameters:
        ----------
        project_scores : dict
            The dictionary containing the scores of the projects of the candidate.

        Returns:
        -------
        cands_dataframe : pandas.DataFrame
            The dataframe containing the project scores of the candidates.
        """

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

                _, name, experience = project.split("__")
                project_names.append(name)
                project_experiences.append(float(experience))

        cands_data_dict = {"id": cand_ids, "name": project_names,
                           "project_score": project_scores_list, "experience": project_experiences}
        # print(cands_data_dict)
        cands_dataframe = pd.DataFrame(cands_data_dict)

        return cands_dataframe

    def __normalize_project_scores(self):
        """
        Normalizes the project scores of the candidates.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

        minimum_mean = self.sim_score_penalty_params['minimum_mean']
        dev_e_factor = self.sim_score_penalty_params['dev_e_factor']

        # Step 1: Get the mean of the column
        project_score_mean = max(
            self.cands_dataframe['project_score'].mean(), minimum_mean)

        # Step 2: Subtract the mean from the column to create a new column
        self.cands_dataframe['project_devs'] = self.cands_dataframe['project_score'] - \
            project_score_mean

        # Step 3: Apply a lambda function on the new column
        self.cands_dataframe['project_devs'] = self.cands_dataframe['project_devs'].apply(
            lambda x: max(round(2-math.exp(-dev_e_factor*x), 2), 0))

        # Step 4: Multiply the first column and the new column and store the value in the first column
        self.cands_dataframe['project_score'] = self.cands_dataframe['project_score'] * \
            self.cands_dataframe['project_devs']

        # Step 5: Delete the new column
        self.cands_dataframe.drop('project_devs', axis=1, inplace=True)

        # Step 6: Apply a lambda function to the first column
        self.cands_dataframe['project_score'] = self.cands_dataframe['project_score'].apply(
            lambda x: round(x*100, 2))

        return 1

    def __drop_irrelevant_projects(self):
        """
        Drops the projects with zero scores.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

        cutoff = self.sim_score_penalty_params['cutoff_score_after_penalty']*100

        temp_dataframe = self.cands_dataframe[self.cands_dataframe['project_score'] >= cutoff]

        if len(temp_dataframe) != 0:
            self.cands_dataframe = temp_dataframe

        return

    def __normalize_experience_scores(self):
        """
        Normalizes the experience scores of the candidates.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

        column_percentiles = np.percentile(
            self.cands_dataframe['experience'], [75, 50, 25])

        # The result will be an array containing the 75th, 50th, and 25th percentiles
        percentile_75th = column_percentiles[0]
        percentile_50th = column_percentiles[1]
        percentile_25th = column_percentiles[2]

        penalty_75_100 = self.experience_penalties['75_100']
        penalty_50_75 = self.experience_penalties['50_75']
        penalty_25_50 = self.experience_penalties['25_50']
        penalty_0_25 = self.experience_penalties['0_25']

        self.cands_dataframe['experience'] = self.cands_dataframe['experience'].apply(lambda x: penalty_75_100 if x >= percentile_75th else (
            penalty_50_75 if x >= percentile_50th else (penalty_25_50 if x >= percentile_25th else penalty_0_25)))

        return 1

    def __create_final_scores_dataframe(self):
        """
        Creates a dataframe containing the final scores of the candidates.
        
        Parameters:
        ----------
        None
        
        Returns:
        -------
        None
        """
        
        self.cands_dataframe['final_score'] = self.cands_dataframe['project_score'] * self.cands_dataframe['experience']
        self.cands_dataframe['final_score'] = self.cands_dataframe['final_score'].apply(
            lambda x: min(round(x, 2), 100))

        self.cands_dataframe['project_count'] = self.cands_dataframe.groupby(
            'id')['final_score'].transform('count')
        self.cands_final_score_dataframe = self.cands_dataframe.groupby('id').agg(
            {'final_score': 'sum', 'project_count': 'first'}).reset_index()
        self.cands_final_score_dataframe['final_score'] = self.cands_final_score_dataframe['final_score'] / self.cands_final_score_dataframe['project_count']

        # print(self.cands_final_score_dataframe)

        return 1

    def __calc_project_count_final_normalized_scores(self):
        """
        Calculates the final normalized scores of the candidates based on the project count.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

        total_entries = len(self.cands_final_score_dataframe)
        self.cands_final_score_dataframe['project_count'] = self.cands_final_score_dataframe['project_count'].apply(
            lambda x: min(x, 5))
        # mode = self.cands_final_score_dataframe['project_count'].mode()[0]

        count_dataframe = pd.DataFrame(
            self.cands_final_score_dataframe['project_count'].value_counts())
        count_dataframe['percentage'] = (
            count_dataframe['count'] / total_entries) * 100

        percentage_list = [0, 0, 0, 0, 0]
        penalties = [1, 1, 1, 1, 1]
        max_percentage = 0
        for index, row in count_dataframe.iterrows():
            percentage_list[index-1] = row['percentage']
            if row['percentage'] >= max_percentage:
                max_percentage = row['percentage']
                mode = index

        if mode == 1:
            if percentage_list[0] >= 50:
                penalties = self.project_count_penalties['mode_1']['more_than_50']
            else:
                penalties = self.project_count_penalties['mode_1']['less_than_50']

        elif mode == 2:
            percentage_345 = sum(percentage_list[2:])
            if percentage_345 > self.project_count_penalties['equivalence_factor']*percentage_list[1]:
                penalties = self.project_count_penalties['mode_2']['3_5_equivalent']
            else:
                penalties = self.project_count_penalties['mode_2']['normal']

        elif mode == 3:
            percentage_45 = sum(percentage_list[3:])
            if percentage_45 > self.project_count_penalties['equivalence_factor']*percentage_list[2]:
                penalties = self.project_count_penalties['mode_3']['4_5_equivalent']
            else:
                penalties = self.project_count_penalties['mode_3']['normal']

        elif mode == 4:
            penalties = self.project_count_penalties['mode_4']

        elif mode == 5:
            penalties = self.project_count_penalties['mode_5']

        self.cands_final_score_dataframe['project_count'] = self.cands_final_score_dataframe['project_count'].apply(
            lambda x: penalties[x-1])
        self.cands_final_score_dataframe['final_score'] = self.cands_final_score_dataframe['final_score'] * \
            self.cands_final_score_dataframe['project_count']
        self.cands_final_score_dataframe['final_score'] = self.cands_final_score_dataframe['final_score'].apply(
            lambda x: round(x, 2))

        self.cands_final_score_dataframe.drop(
            'project_count', axis=1, inplace=True)

        return

    def __get_score_reasons_and_personality_scores(self, candidate_recruit_answers, candidate_score, candidate_description):
        candidate_description, candidate_tech_skills = candidate_description.split("SKILLS: ")

        system_prompt = """You are a reasoning agent who is trying to help a company find best candidates for recruitment and reasons out why a particular candidate is suitable or unsuitable for a particular job based on the candidate's past projects and skills. Aditionally, you also judge an applicant's personality and his/her willingness to go to Japan to work for the company."""

        user_prompt = f"""We have job description for a job position in the field of technology.
        Multiple candidates applied for the job. All of them submitted their resumes and we have calculated a score that shows the aptness of the applicant for the job position. You are supposed to carry out the following 2 tasks.

        TASK-1: Providing a reasoning for the suitability/unsuitability of the applicant for the job based on his/her projects and skills

        We will give you a job description and the set of projects of the applicant alongwith the score that we calculated. You have to analyse the job description, the projects, and provide a reasoning for why the applicant has been given that score.

        The score is given out of 100. An applicant may get a high, low, or a moderate score. So carefully analyze the job description, the projects and then provide a reasoning as to why the applicant has a particular score. Say an applicant has a bad score then you need to justify how the applicant is not so well suited for the job based on the job description and the applicant's projects. Similarly if the applicant has a high score then you need to provide a reasoning as to why the applicant is suited for the job.

        Furthermore, the resume also contains the skills of the applicant. Alongwith the job description, the company has also provided us with a list of skills that it requires the applicants to have. You will be given a list of both the skills, the applicant skills as well as the skills required by the company. Your task is to list out the skills that are common in both alongwith the skills that are required but are not possessed by the applicant. You have to provide a reasoning based on those skills that are required but are not possessed by the applicant.

        The job description provided by the company: {self.jdk_desc}

        The applicant has worked on the following projects: {candidate_description}.

        The applicant has been given a score of {candidate_score}.

        Required skills set shared by the company: {self.jdk_tech_skills}.
        Applicant skills: {candidate_tech_skills}.
        
        The above are a few skills mentioned in each candidates description. And a few skills mentioned in the job description that are requires skills. You have to find out the common skills in the required skills and the candidate skills, and the skills that are required but the candidate lack and finally provide a reasoning based on those skills along with the previously given project andf job descriptions. Also mention those skills with a tag of common and absent respectively. You do not need to justify all the extra skills of the candidate to be relevant to the job description. You only need to justify the skills that are required for the job but the candidate lacks.

        You have to return the output in the following format. Remember to be very brief while providing the reasoning. Try not to exceed 60 words.
        
        TASK-2: Determining the applicant's personality and his/her willingness to work in Japan based on the answers given by him/her

        As mentioned earlier, one of your jobs is to determine the applicant's personality and his/her willingness to work in Japan. To do this task, you will be given a JSON containing all the questions which were asked to the candidate along with the answers that the applicant gave. You have to give a single score based on the applicant's personality and his/her willingness to go to Japan.

        While judging the personality of the applicant, you also have to consider the soft skills that the company is looking for in a candidate. Be sure to consider those skills as they are very important for the company.

        The soft skills that the company is looking for are: {self.jdk_soft_skills}.

        The question answers given by the applicant: {candidate_recruit_answers}.""" + """
        
        You have to return the output of all the above 3 tasks in the following JSON format:
            tech_reason: <GIVE THE REASONING HERE THAT IS ASKED FOR IN TASK-1>,
            score: <GIVE A SCORE OUT OF 5 HERE FOR TASK-2>,
            reason: <GIVE A REASON FOR THE SCORE YOU GAVE IN TASK-2>

        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        # print(response)

        response = json.loads(response.choices[0].message.content, strict=False)

        techreason = response['tech_reason']
        score = max(min(response['score'], 5), 0)
        reason = response['reason']

        return techreason, score, reason

    def __add_reasons_and_scores(self):
        """
        Returns the personality score of the candidate.

        Returns
        -------
        int
            The personality score of the candidate.
        """

        for index, row in self.cands_final_score_dataframe.iterrows():
            candidate_id = row['id']
            candidate_recruit_answers = self.candidate_recruit_answers[self.candidate_id_list.index(
                candidate_id)]

            candidate_score = row['final_score']
            candidate_projects_info = self.candidate_desc_list[self.candidate_id_list.index(
                candidate_id)]


            techreason, score, reason = self.__get_score_reasons_and_personality_scores(candidate_recruit_answers, candidate_score, candidate_projects_info)

            self.cands_final_score_dataframe.loc[index, 'tech_reason'] = techreason
            self.cands_final_score_dataframe.loc[index, 'personality_score'] = score
            self.cands_final_score_dataframe.loc[index, 'personality_reason'] = reason

        return

    def __get_all_candidate_scores(self):
        """

        """
        project_scores_all = {}

        # abs_start_time = time.time()
        # start_time = time.time()
        for candidate_id in self.candidate_id_list:
            project_scores = self.__fetch_candidate_scores(candidate_id)
            project_scores_all[candidate_id] = project_scores
        # print(project_scores_all)
        # print("Data Fetch Pinecone --- %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()
        self.cands_dataframe = self.__create_dataframe(project_scores_all)
        # print(self.cands_dataframe)
        self.__normalize_project_scores()
        # print(self.cands_dataframe)
        self.__drop_irrelevant_projects()
        self.__normalize_experience_scores()
        self.__create_final_scores_dataframe()
        self.__calc_project_count_final_normalized_scores()
        # print("Scoring Done --- %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()
        # self.__add_cand_score_reasons()
        # print("Reasoning Done --- %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()
        self.__add_reasons_and_scores()
        # print("Personality Done --- %s seconds ---" % (time.time() - start_time))
        pd.DataFrame.to_excel(self.cands_dataframe, f"./results/{self.jdk_id}.xlsx", index=False)
        pd.DataFrame.to_excel(self.cands_final_score_dataframe, f"./results/jdk_{self.jdk_id}.xlsx", index=False)

        result_data_json = self.cands_final_score_dataframe.to_json(
            orient='records')
        # print("JSON Done --- %s seconds ---" % (time.time() - start_time))
        # print("Total Time --- %s seconds ---" % (time.time() - abs_start_time))

        return result_data_json

    def score_candidates(self):
        """
        Scores the candidates based on their projects and the job description.

        Parameters:
        ----------
        None

        Returns:
        -------
        candidate_scores : list
            The list of dictionaries containing the id and the score of the candidates.
        """

        self.jdk_embedding = self.__fetch_jdk_embeddings()
        candidate_scores = self.__get_all_candidate_scores()
        return candidate_scores


def get_candidate_scores(jdk_info, candidates_info):
    """
    Scores the candidates based on their projects and the job description.

    Parameters:
    ----------
    jdk_info : dict
        The dictionary containing the id and the description of the job description.
    candidates_info : list
        The list of dictionaries containing the id and the description of the candidates.

    Returns:
    -------
    result : list
        The list of dictionaries containing the id and the score of the candidates.
    """

    _ = load_dotenv(find_dotenv())

    # print(os.environ.get('PINECONE_API_KEY'))
    jdk_resume_assistant = HRAssistant(jdk_info, candidates_info)
    result = jdk_resume_assistant.score_candidates()

    return result
