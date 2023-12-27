import os
import openai

# Load .env file
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Importing modules for langchain
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser, JsonKeyOutputFunctionsParser


# Pydantic models
# Project model
class Project(BaseModel):
    """A project with a name and description."""

    name: str = Field(description="The name of the project.")
    relevance: float = Field(description="The relevance of the project to the job description.")
    # reason: bool = Field(description="relation of project, true if directly relevant, false otherwise")
    reason: str = Field(description="relation of project, directly relevant, indirectly relevant or not relevant")
    final: bool = Field(description="true if directly relevant, false otherwise")

# Experience model
class Experience(BaseModel):
    """Projects that a candidate has worked on."""

    projects: List[Project] = Field(
        description="A list of projects that the candidate has worked on."
    )


# OpenAI model
model = ChatOpenAI(model='gpt-3.5-turbo' ,temperature=0)


# Convert Pydantic model to OpenAI function
extraction_functions = [convert_pydantic_to_openai_function(Experience)]
extraction_model = model.bind(functions=extraction_functions, function_call={"name": "Experience"})


# Template
# template = """A job description will be passed to you along with a candidate's project experience.

# You will be asked to extract the projects very strictly relevant to the job description.

# If no project mentioned are relevant it's fine - you don't need to extract any! Just return an empty list.

# Do not make up any new project. Strictly return the relevant projects only.

# The company's jdk is as follows:"""

template = """You are a resume matching agent. You will be given a job description for a job in the field of technology.

There are multiple applicants for the job and all of them have sent in their resumes. Each of the resumes has multiple number of projects. You will be given the individual applicant's projects' details.

Your task is to give the relevance score to each of the projects with respect to the job description. The relevance score refers to how relevant a project is to the job description. The relevance score should be between 0 and 1 with 0 being the least relevance score and 1 being the maximum relevance score.

Make sure to mark the provided projects only and not create some additional projects.

The company's job description is as follows:"""

# Company's job description
# jdk = """Data analysis and applicability study of algorithm using AI technologies such as machine learning, deep learning, etc. Study and evaluate AI algorithm and NTT Laboratories research products including log analysis AI of telecommunication network, image analysis AI, and so on. For example, trainee is expected to work on the following items:  ++ AI implementation (using machine learning, deep learning, etc.) with Python ++ Study/evaluation of certain algorithms' applicability ++ Analysis and visualization of those results (MS Excel, BI tool)"""
jdk = """Join our dynamic team as a Web Development Intern. In this role, you'll work closely with our experienced developers to contribute to the design, coding, testing, and deployment of web applications. This internship offers hands-on experience with HTML, CSS, and JavaScript, as well as exposure to popular web development frameworks. It would be great if the candidate has previous experience with web development"""

# Chat prompt
# prompt = ChatPromptTemplate.from_messages([
#     ("system", template + "\n" + jdk),
#     ("user", "What projects are strictly relevant to the company's requirements?"),
#     ("human", "{input}")
# ])
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a recruiter searching for suitable candidates for your companies job requirement. The job description is as follows" + "\n" + jdk),
    ("user", "What projects are strictly relevant to the company's requirements?"),
    ("human", "{input}")
])


# Chain
# extraction_chain = prompt | extraction_model | JsonOutputFunctionsParser()
# extraction_chain = prompt | extraction_model
extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="projects")


# Project experience
project_prompt = """I have worked on the following projects:

1. Literature Society IITJ Website:
    A website developed for managing various activities for literature society of IIT Jodhpur. The website contains the following features:
        -User Profile Management (with in site notification service)
        -Library Management System
        -Feature to upload written works to be displayed on website (Reader's Section).
        -Moderator and Admin access profiles to approve books and contents.

2. Multi Model Data Analysis for Annotation of Human Activities:
    -Performed data analysis and outlier detection on a dataset of accelerometer and gyroscope readings.
    -Ideated and executed a pipeline for performing cleaning and detecting the activity type throught the dataset.

3. Cloudphysician's Vital Extraction Challenge:
    -Fine-tuned DETR and detectron to for ICU monitor segmentation images of ICU rooms and vital segmentation from ICU monitors.
    -Worked on graph digitisation algorithm to digitize ECG strip.

4. Large Language Models Post-Processing module:
    -Developed pipeline for bias/toxicity detection in outputs of large language models and integrated GPT-2, BERT and other language models for testing.
    -Developed user-friendly UI (ReactJS) and backend (Django Rest Framework) for integration of detectors and providing an intuitive platform for using the developed pipeline.
"""


# Response
response = extraction_chain.invoke({"input": project_prompt})

print(response)