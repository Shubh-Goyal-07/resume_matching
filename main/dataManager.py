import openai
from pinecone import Pinecone
from google.cloud import translate_v2 as translate

from dotenv import load_dotenv, find_dotenv
import os

from datetime import datetime
import json


class Manager_model():
    """
    This class is used to add a new candidate or a new job description to the database.

    ...

    Attributes
    ----------
    data : dict
        A dictionary containing the raw data of the candidate or the job description.
    prompt : PromptTemplate
        A PromptTemplate object that is used to generate the final (concise and precise) description of the candidate or the job description.
    max_experience_days : int
        The maximum number of days of experience that can be added to the final description of the candidate.

    Methods
    -------
    __create_jdk_prompt()
        Creates a PromptTemplate object for getting the final description of the job description.
    __create_candidate_prompt()
        Creates a PromptTemplate object for getting the final description of the candidate.
    __get_final_description(title, description, skills)
        Returns the final description of the candidate or the job description using the OpenAI LLM model (GPT-3.5-turbo).
    __upsert_to_database(namespace, embeddings_vector)
        Upserts the given vector to the given namespace in the database.
    __get_embeddings(description_list)
        Returns the embeddings of the gicen list of descriptions using the OpenAIEmbeddings model (ada-v2).
    __get_jdk_final_description(title, description, skills)
        Returns the final description of the job description.
    __get_cand_combined_desc(titles, final_descriptions, skill_info)
        Returns the combined description of all the projects of the candidate.
    __get_experience(start_date, end_date)
        Returns the experience of the candidate in months.
    __get_personality_score()
        Returns the personality score of the candidate.
    add_jdk()
        Adds the job description to the database.
    add_candidate(save_gen_desc_only=False)
        Adds the candidate to the database.
    """

    def __init__(self):
        """
        Parameters
        ----------
        data : dict
            A dictionary containing the raw data of the candidate or the job description.
        """

        _ = load_dotenv(find_dotenv())

        openai.api_key = os.environ.get("OPENAI_API_KEY")

        self.client = openai.OpenAI()

        self.pc = Pinecone(
            api_key=os.environ.get('PINECONE_API_KEY'),
            environment='gcp-starter'
        )

        config = json.load(open('./config.json'))
        self.max_experience_days = config['experience_params']['maximum_experience'] * 30

        self.pinecone_config = config['pinecone_config']

    def __translate_ja_en(self, description):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"../google-credentials.json"

        translate_client = translate.Client()
        target = "en"

        output = translate_client.translate(description, target_language=target)

        return output['translatedText']

    def __create_jdk_prompt(self, title, description, skills):
        """
        Creates a PromptTemplate object for getting the final description of the job description.

        Returns
        -------
        None
        """

        system_prompt = """You are a summarizing agent who takes job descriptions and converts them into concise and precise two line descriptions."""

        user_prompt = f"""We have jobs to offer to interested people. Each job has a title, a job description, and the skills required for that job. To help people understand the job position in a better manner, your task is to combine all these three separate things to create a concise and more precise description of the job. Remove irrelevant information from the job description and only include the important information that will help people understand the job position better.

        Job Title: {title}
        Job Description: {description}
        Skills Required: {skills}
        """

        return system_prompt, user_prompt

    def __create_candidate_prompt(self, title, description, skills):
        """
        Creates a PromptTemplate object for getting the final description of the project description of the candidate.

        Returns
        -------
        None
        """

        system_prompt = """You are a summarizing agent who takes project descriptions and converts them into a more concise and precise two line descriptions."""

        user_prompt = f"""All the candidates who have applied for jobs have submiited their resumes. All resumes contain the details about the projects that the candidate has worked on. Each project has a title, a description, and the skills that were put to use to complete that project. Your task is to combine all these three separate things to create a concise and more precise description of the project. Remove irrelevant information from the project description and only include the important information that will be relevant when applying for a job.
        
        Project Title: {title}
        Project Description: {description}
        Skills Used: {skills}
        """

        return system_prompt, user_prompt

    def __get_final_description(self, system_prompt, user_prompt):
        """
        Returns the final description of the candidate or the job description using the OpenAI LLM model (GPT-3.5-turbo).

        Parameters
        ----------
        title : str
            The title of the candidate's project or the job description.
        description : str
            The description of the candidate's project or the job description.
        skills : str
            The skills required for the job or the skills used in the project.

        Returns
        -------
        str
            The final description of the candidate or the job description.
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        final_description = response.choices[0].message.content

        return final_description

    def __upsert_to_database(self, namespace, embeddings_vector):
        """
        Upserts the given vector to the given namespace in the database.

        Parameters
        ----------
        namespace : str
            The namespace to which the vector has to be upserted.
        embeddings_vector : list
            The vector to be upserted.

        Returns
        -------
        None     
        """

        index = self.pc.Index(self.pinecone_config['index_name'])

        index.upsert(
            vectors=embeddings_vector,
            namespace=namespace
        )

        return

    def __get_embeddings(self, description_list):
        """
        Returns the embeddings of the given list of descriptions using the OpenAIEmbeddings model (ada-v2).

        Parameters
        ----------
        description_list : list
            The list of descriptions whose embeddings have to be calculated.

        Returns
        -------
        list
            The list of embeddings of the given descriptions.
        """

        embeddings = self.client.embeddings.create(input=description_list, model="text-embedding-ada-002").data
        embeddings = [element.embedding for element in embeddings]
        # print(embeddings)

        return embeddings

    def __get_jdk_final_description(self, title, description, skills):
        """
        Returns the final description of the job description.

        Parameters
        ----------
        title : str
            The title of the job description.
        description : str
            The description of the job description.
        skills : str
            The skills required for the job.

        Returns
        -------
        str
            The final description of the job description.
        """

        description = f"{description}" + f"SKILLS: {skills}"

        return description

    def add_jdk(self, data):
        """
        Adds the job description to the database.

        Returns
        -------
        str
            The final description of the job description.
        """
        self.data = data

        jdk_id = self.data['id']
        title = self.data['title']
        description = self.data['description']
        skills = self.data['skills']
        skills = ", ".join(skills)

        description = self.__translate_ja_en(description)
        system_prompt, user_prompt = self.__create_jdk_prompt(title, description, skills)
        final_description = self.__get_final_description(system_prompt, user_prompt)

        embeddings = self.__get_embeddings([final_description])

        description_embeddings = embeddings[0]

        jdk_namespace = self.pinecone_config['jdk_namespace']

        vector = [{"id": str(jdk_id), "values": description_embeddings, "metadata": {"jdk_id": str(jdk_id)}}]

        self.__upsert_to_database(jdk_namespace, vector)

        final_description = self.__get_jdk_final_description(
            title, final_description, skills)

        return final_description

    def __get_cand_combined_desc(self, titles, final_descriptions, skill_info):
        """
        Returns the combined description of all the projects of the candidate.

        Parameters
        ----------
        titles : list
            The list of titles of the candidate's projects.
        final_descriptions : list
            The list of descriptions of the candidate's projects.
        skill_info : list
            The list of skills used in the candidate's projects.

        Returns
        -------
        str
            The combined description of all the projects of the candidate.
        """

        all_project_desc = ""

        for title, description, skills in zip(titles, final_descriptions, skill_info):
            all_project_desc += f"The project is titled '{title}'. {description}" + f"The project uses {skills}. "

        return all_project_desc

    def __get_experience(self, start_date, end_date):
        """
        Returns the experience of the candidate in months.

        Parameters
        ----------
        start_date : str
            The start date of the candidate's project.
        end_date : str
            The end date of the candidate's project.

        Returns
        -------
        int
            The experience of the candidate in months.
        """

        date_format = "%d/%m/%Y"

        end_date_obj = datetime.strptime(end_date, date_format).date()
        start_date_obj = datetime.strptime(start_date, date_format).date()

        exp_diff = end_date_obj - start_date_obj

        experience = min(exp_diff.days, self.max_experience_days) / 15
        experience = int(round(experience, 0))

        return experience

    def add_candidate(self, data):
        """
        Adds the candidate to the database.

        Parameters
        ----------
        data : dict
            A dictionary containing the raw data of the candidate.

        Returns
        -------
        string
            A string containing the final description of the candidate.
        """

        self.data = data

        candidate_id = self.data['id']
        projects = self.data['projects']


        titles = []
        actual_titles = []
        final_descriptions = []
        skill_info = []
        unique_skills = set()
        metadatas = []

        for project in projects:
            title = project['title']
            description = project['description']
            skills = project['skills']
            unique_skills.update(skills)
            skills = ", ".join(skills)

            end_date = project['endDate']
            start_date = project['startDate']
            experience = self.__get_experience(start_date, end_date)

            titles.append(f"{candidate_id}__{title}")
            actual_titles.append(f"{title}")

            system_prompt, user_prompt = self.__create_candidate_prompt(title, description, skills)
            final_description = self.__get_final_description(system_prompt, user_prompt)
            final_descriptions.append(final_description)

            metadatas.append({"candidate_id": f'{candidate_id}', 'experience': experience})

            skill_info.append(skills)

        all_projects_desc = self.__get_cand_combined_desc(
            actual_titles, final_descriptions, skill_info)
        all_projects_desc = f"The candidate has worked on the following projects: {all_projects_desc}" + f"SKILLS:  {', '.join(unique_skills)}."

        final_descriptions.append(all_projects_desc)
        embeddings = self.__get_embeddings(final_descriptions)

        namespace = self.pinecone_config['projects_namespace']

        description_vector = [
            {"id": title, "values": embeddings[i], 'metadata': metadatas[i]} for i, title in enumerate(titles)]
        self.__upsert_to_database(namespace, description_vector)

        gen_desc_vec = [{"id": f"{candidate_id}", "values": embeddings[-1]}]
        cand_desc_namespace = self.pinecone_config['candidate_description_namespace']
        self.__upsert_to_database(cand_desc_namespace, gen_desc_vec)

        return all_projects_desc

    def delete_candidate(self, data):
        index = self.pc.Index(self.pinecone_config['index_name'])

        # deleting general description embedding
        index.delete(
            ids=[str(self.data['id'])],
            namespace=self.pinecone_config['candidate_description_namespace']
        )

        self.data = data

        # deleting project embeddings
        projects = self.data['projects']
        titles = [f"{self.data['id']}__{project}" for project in projects]

        index.delete(
            ids=titles,
            namespace=self.pinecone_config['projects_namespace']
        )

        return

    def delete_jdk(self, data):
        index = self.pc.Index(self.pinecone_config['index_name'])

        self.data = data

        index.delete(
            ids=[str(self.data['id'])],
            namespace=self.pinecone_config['jdk_namespace']
        )

        return


def upsert_to_database(category, data):
    """
    Adds the candidate or the job description to the database.

    Parameters
    ----------
    category : str
        The category of the data. It can be either "candidate" or "jdk".
    data : dict
        A dictionary containing the raw data of the candidate or the job description.
    save_gen_desc_only : bool, optional
        A boolean value that indicates whether to save only the generated description of the candidate or to save the generated description along with the descriptions of the candidate's projects. The default value is False.

    Returns
    -------
    str
        The final description in case of "jdk" category and a dictionary containing the final description of the candidate and the personality score of the candidate in case of "candidate" category.
    """

    # Load .env file
    _ = load_dotenv(find_dotenv())
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    pc = Pinecone(
        api_key=os.environ.get('PINECONE_API_KEY'),
        environment='gcp-starter'
    )

    upsert_model = Manager_model()

    if category == "candidate":
        description = upsert_model.add_candidate(data)
    else:
        description = upsert_model.add_jdk(data)

    return description
