import os
import openai

# Importing modules for langchain
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser, JsonKeyOutputFunctionsParser


# PYDANTIC MODELS
# Project model
class Project(BaseModel):
    """A project's name, relevance and reason according to a given job description."""

    name: str = Field(description="The title of the project.")
    relevance: int = Field(description="Strictly Necessary. The score of the project based on the job description. 1 being the lowest and 10 being the highest.")
    # reason: bool = Field(description="relation of project, true if directly relevant, false otherwise")
    reason: str = Field(description="The logical reason behind the relevance or irrelevance of the project.")
    # final: bool = Field(description="true if directly relevant, false otherwise")

# Experience model
class Experience(BaseModel):
    """Projects that a candidate has worked on."""

    projects: List[Project] = Field(
        description="A list of projects that the candidate has worked on."
    )


class JDK_projects():
    def __init__(self, jdk, resume):
        self.jdk = jdk
        self.resume = resume


    def __create_model(self):
        # OpenAI model
        model = ChatOpenAI(model='gpt-3.5-turbo' ,temperature=0)

        # Convert Pydantic model to OpenAI function
        extraction_functions = [convert_pydantic_to_openai_function(Experience)]
        extraction_model = model.bind(functions=extraction_functions, function_call={"name": "Experience"})

        return extraction_model
    

    def __create_prompt(self):
        # Template
        # template = """A job description will be passed to you along with a candidate's project experience.

        # You will be asked to extract the projects very strictly relevant to the job description.

        # If no project mentioned are relevant it's fine - you don't need to extract any! Just return an empty list.

        # Do not make up any new project. Strictly return the relevant projects only.

        # The company's jdk is as follows:"""

        template_system = """You are an experienced recruiter searching for strictly suitable candidates for your company's current job requirement. The job description is as follows:"""
        jdk_description = self.jdk['description']

        template_user = """Extract the projects very strictly relevant to the job description. If no project mentioned are relevant it's fine - you don't need to extract any! Just return an empty list. Do not make up any new project. Strictly return the relevant projects only."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", template_system + "\n" + jdk_description),
            ("user", template_user),
            ("human", "{input}")
        ])

        return prompt
    

    def __create_project_prompt_template(self):
        projects = self.resume['projects']

        project_prompt = """I have worked on the following projects:"""

        num = 1
        for project in projects:
            project_prompt += f"{num}. {project['name']} \n {project['description']}"
            num += 1

        return project_prompt
    

    def __create_extration_chain(self, extraction_model, prompt):
        extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="projects")
        return extraction_chain
    

    def match_projects(self):
        extraction_model = self.__create_model()
        prompt = self.__create_prompt()
        extraction_chain = self.__create_extration_chain(extraction_model, prompt)

        # Project experience
        project_prompt = self.__create_project_prompt_template()

        # Extract projects
        project_rating = extraction_chain.invoke({"input": project_prompt})

        # Print projects
        # print(project_rating)

        return project_rating
    

def match_projects(jdk, resumes):
    # Load .env file
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv())
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    resume_scores = []
    for resume in resumes:
        jdk_projects = JDK_projects(jdk, resume)
        scores = jdk_projects.match_projects()
        resume_scores.append({'id': resume['id'], 'scores': scores})

    return resume_scores