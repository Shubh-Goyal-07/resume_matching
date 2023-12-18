import json
from jdk_project_match import match_projects
from skill_match import match_skills
from save_scores import save_to_excel

def jdk_resume_loader(jdks_path, resumes_path):
    jdks = json.load(open(jdks_path))
    jdk_abstract = []
    for jdk in jdks:
        id = jdk['id']
        title = jdk['title']
        description = jdk['description']
        skills = jdk['skills']

        jdk_abstract.append({'id': id, 'title': title, 'description': description, 'skills': skills})

    resumes = json.load(open(resumes_path))
    resume_abstract = []
    for resume in resumes:
        id = resume['id']
        name = resume['name']
        skills = resume['skills']
        projects = resume['projects']

        resume_abstract.append({'id': id, 'name': name, 'skills': skills, 'projects': projects})

    return jdk_abstract, resume_abstract


def make_cand_pro_dic(resumes):
    candidate_project_dic = {}
    for resume in resumes:
        candidate_project_dic[resume['id']] = {}

        for project in resume['projects']:
            candidate_project_dic[resume['id']][project['name']] = {'skills': project['skills'], 'experience': project['experience']}

    return candidate_project_dic


def add_relevance_score(project_scores, candidate_project_dic):
    for candidate in project_scores:
        print(candidate)
        for project in candidate['scores']:
            if project['name'] in candidate_project_dic[candidate['id']]:
                # if project['relevance'] > 4:
                # print(project)
                candidate_project_dic[candidate['id']][project['name']]['relevance_score'] = project['relevance'] if project['relevance'] > 4 else 0

    return candidate_project_dic


def resume_scorer(jdk_path, resume_path):
    jdks, resumes = jdk_resume_loader(jdk_path, resume_path)
    candidate_project_dic = make_cand_pro_dic(resumes)
    # print(candidate_project_dic)
    jdk_project_scores = []     # just project scores
    jdk_resume_scores = []      # based on weighted skill scores (final score)

    for jdk in jdks:
        print(f"Matching jdk {jdk['id']}")
        jdk_id = jdk['id']
        jdk_skills = jdk['skills']

        project_scores = match_projects(jdk, resumes)

        jdk_project_scores.append({'jdk_id': jdk_id, 'project_scores': project_scores})
        candidate_project_dic = add_relevance_score(project_scores, candidate_project_dic)
        
        # candidate_scores = match_skills(jdk_skills, candidate_project_dic)

        # jdk_resume_scores.append({'jdk_id': jdk_id, 'candidate_scores': candidate_scores})

        # save_to_excel(jdk_id, candidate_scores)


    json.dump(jdk_project_scores, open('./results/jdk_project_scores.json', 'w'), indent=4)
    # json.dump(jdk_resume_scores, open('./results/jdk_resume_scores.json', 'w'), indent=4)


if __name__ == '__main__':
    resume_scorer('./data/jdks.json', './data/resumes.json')