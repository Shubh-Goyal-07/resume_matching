import json
from jdk_project_match import match_projects

def jdk_resume_loader(jdk_path, resume_path):
    jdks = json.load(open(jdk_path))
    jdk_abstract = []
    for jdk in jdks:
        id = jdk['id']
        title = jdk['title']
        description = jdk['description']

        jdk_abstract.append({'title': title, 'description': description, 'id': id})

    resumes = json.load(open(resume_path))
    resume_abstract = []
    for resume in resumes:
        id = resume['id']
        skills = resume['skills']
        projects = resume['projects']

        resume_abstract.append({'id': id, 'skills': skills, 'projects': projects})

    return jdk_abstract, resume_abstract


def resume_scorer(jdk_path, resume_path):
    jdks, resumes = jdk_resume_loader(jdk_path, resume_path)
    jdk_resume_scores = []

    for jdk in jdks:
        jdk_id = jdk['id']

        resume_scores = match_projects(jdk, resumes)

        jdk_resume_scores.append({'jdk_id': jdk_id, 'resume_scores': resume_scores})

    json.dump(jdk_resume_scores, open('jdk_resume_scores.json', 'w'), indent=4)


if __name__ == '__main__':
    resume_scorer('./data/jdks.json', './data/resumes.json')