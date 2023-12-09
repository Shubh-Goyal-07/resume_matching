# import transformers
# import torch
import torch.nn.functional as F
from numpy import dot
from numpy.linalg import norm


# tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
# model = transformers.BertModel.from_pretrained('bert-base-uncased')

def extract_skills(projects, skills):
    skill_weight_map = {}
    n_skills = len(skills)
    n_projects = len(projects)

    for i in range(n_skills):
        if n_skills[i] not in skill_weight_map.keys():
            skill_weight_map[n_skills[i]] = 0
    
    for i in range(n_projects):
        n_proj_skills = len(n_projects['skills_used'])
        for j in range(n_proj_skills):
            if n_proj_skills[j] in skill_weight_map.keys():
                skill_weight_map[n_proj_skills[j]] += 1
            else: skill_weight_map[n_proj_skills[j]] = 1

import os
from openai import OpenAI

# print(response)

def get_skill_embeddings(skill):

#   inputs = tokenizer.encode_plus(skill,  
#                           add_special_tokens=True,
#                           return_tensors='pt')
  
#   outputs = model(**inputs)
  
#   last_hidden_states = outputs[0][:, 0, :]
  
#   return last_hidden_states
    client = OpenAI()
    client.api_key = os.getenv("OPENAI_API_KEY")
    response = client.embeddings.create(input = [skill], model="text-embedding-ada-002").data[0].embedding
    return response

job_skills = [
    "Java11",
    "Bootstrap4.3",
    "MySQL8.0.16",
]

# Sample skills of the applicant (replace with your actual data)
applicant_skills = [
    "Python",
    "Machine",
    "MySQL",
    "Java",
]

# Sample skill-to-weight mapping dictionary for the applicant (replace with your actual data)
skill_weights = {
    "Java": 0,
    "Angular": 0.9,
    "Python": 0.3,
    "Data analysis": 0.6,
    "Machine learning": 0.5,
}

def calculate_similarity(skill1, skill2):
    tensor1 = get_skill_embeddings(skill1)
    tensor2 = get_skill_embeddings(skill2)
    # tensor1 = tensor1.view(1, -1)
    # tensor2 = tensor2.view(1, -1)

    # Calculate cosine similarity
    # return F.cosine_similarity(tensor1, tensor2)
    cos_sim = dot(tensor1, tensor2)/(norm(tensor1)*norm(tensor2))
    return cos_sim


print('hereeeeeeeee1')
# Calculate similarity scores for all pairs of skills
similarity_scores = []
for job_skill in job_skills:
    max_val = 0
    a_skill = ''
    for applicant_skill in applicant_skills:
        similarity = calculate_similarity(job_skill, applicant_skill)
        # similarity_scores.append((job_skill, applicant_skill, weighted_similarity))
        if similarity>max_val:
            max_val = similarity
            a_skill = applicant_skill
    similarity_scores.append([job_skill, a_skill, max_val])

print("hereeeeeeeeeee2")
print(similarity_scores)
n = len(similarity_scores)
sim_score = 0
for i in range(n):
    sim_score += skill_weights[similarity_scores[i][1]]*similarity_scores[i][2]

# Print similarity scores
print(sim_score)