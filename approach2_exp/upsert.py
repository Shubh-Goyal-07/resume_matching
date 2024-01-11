from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
import openai
import pinecone

from dotenv import load_dotenv, find_dotenv
import os

from datetime import datetime
import json


class Upsert_model():
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

    def __init__(self, data):
        """
        Parameters
        ----------
        data : dict
            A dictionary containing the raw data of the candidate or the job description.
        """

        self.data = data

        config = json.load(open('./config.json'))
        self.max_experience_days = config['experience_params']['maximum_experience'] * 30

    def __create_jdk_prompt(self):
        """
        Creates a PromptTemplate object for getting the final description of the job description.
        
        Returns
        -------
        None
        """
        
        template = """Job Title: {title}
        Job Description: {description}
        
        Combine the above information to create a concise and more precise description of the job. Remove irrelevant information from the job description and only include the important information. Convert the job description into a project description. Wrap it in two sentences."""

        self.prompt = PromptTemplate(template=template, input_variables=[
                                     "title", "description"])

        return

    def __create_candidate_prompt(self):
        """
        Creates a PromptTemplate object for getting the final description of the project description of the candidate.

        Returns
        -------
        None
        """

        template = """Project Title: {title}
        Project Description: {description}
        
        Combine the above information to create a concise and more precise description of the project. Remove irrelevant information from the project description and only include the important information. Wrap it in two sentences."""

        self.prompt = PromptTemplate(template=template, input_variables=[
                                     "title", "description"])

        return

    def __get_final_description(self, title, description, skills):
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

        llm = OpenAI(model="gpt-3.5-turbo-instruct")
        llm_chain = LLMChain(prompt=self.prompt, llm=llm)

        final_description = llm_chain.run(title=title, description=description)

        final_description = f"{final_description} Skills used includes {skills}."

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

        index = pinecone.Index("willings")

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

        embedding_model = OpenAIEmbeddings()
        embeddings = embedding_model.embed_documents(description_list)

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

        description = f"The job title is {title}, and the description is as follows: {description}. The skills required for the job are {skills}."

        return description

    def add_jdk(self):
        """
        Adds the job description to the database.

        Returns
        -------
        str
            The final description of the job description.
        """

        jdk_id = self.data['id']
        title = self.data['title']
        description = self.data['description']
        skills = self.data['skills']
        skills = ", ".join(skills)

        self.__create_jdk_prompt()
        final_description = self.__get_final_description(
            title, description, skills)

        embeddings = self.__get_embeddings([final_description])

        description_embeddings = embeddings[0]

        vector = [{"id": str(jdk_id), "values": description_embeddings}]

        self.__upsert_to_database("jdks", vector)
        description = self.__get_jdk_final_description(
            title, final_description, skills)

        return description

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
            all_project_desc += f"The project is titled '{title}'. {description} The project uses {skills}. "

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

    def add_candidate(self, save_gen_desc_only=False):
        """
        Adds the candidate to the database.

        Parameters
        ----------
        save_gen_desc_only : bool, optional
            A boolean value that indicates whether to save only the generated description of the candidate or to save the generated description along with the descriptions of the candidate's projects. The default value is False.
        
        Returns
        -------
        dict
            A dictionary containing the final description of the candidate and the personality score of the candidate.
        """

        candidate_id = self.data['id']
        projects = self.data['projects']

        self.__create_candidate_prompt()

        titles = []
        actual_titles = []
        final_descriptions = []
        skill_info = []

        for project in projects:
            title = project['title']
            description = project['description']
            skills = project['skills']
            skills = ", ".join(skills)

            end_date = project['endDate']
            start_date = project['startDate']
            experience = self.__get_experience(start_date, end_date)

            titles.append(f"{title}__{experience}")
            actual_titles.append(f"{title}")

            final_description = self.__get_final_description(
                title, description, skills)
            final_descriptions.append(final_description)

            skill_info.append(skills)

        all_projects_desc = self.__get_cand_combined_desc(
            actual_titles, final_descriptions, skill_info)

        if save_gen_desc_only:
            embeddings = self.__get_embeddings([all_projects_desc])
        else:
            final_descriptions.append(all_projects_desc)
            embeddings = self.__get_embeddings(final_descriptions)
            namespace = f"candidate_{candidate_id}"
            description_vector = [
                {"id": title, "values": embeddings[i]} for i, title in enumerate(titles)]
            self.__upsert_to_database(namespace, description_vector)

        gen_desc_vec = [{"id": f"{candidate_id}", "values": embeddings[-1]}]
        namespace_all = "all_candidates"
        self.__upsert_to_database(namespace_all, gen_desc_vec)

        return all_projects_desc


def upsert_to_database(category, data, save_gen_desc_only=False):
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

    pinecone.init(
        api_key=os.environ.get('PINECONE_API_KEY'),
        environment='gcp-starter'
    )

    upsert_model = Upsert_model(data)

    if category == "candidate":
        description = upsert_model.add_candidate(
            save_gen_desc_only=save_gen_desc_only)
    else:
        description = upsert_model.add_jdk()

    return description
