from dataManager import upsert_to_database
from hrAssistant import get_candidate_scores
from candAssistant import get_job_suggestions
from dataManager import Manager_model
import json
import time


jdks = json.load(open('../data/jdks.json'))
resumes = json.load(open('../data/resumes.json'))

abs_start_time = time.time()

# print("Upserting jdks to database...")
# start_time = time.time()
# for jdk in jdks:
#     desc = upsert_to_database("jdk", jdk)
#     # json.dump({'id': jdk['id'], 'description':desc}, open(f"../new_data/jdks/{jdk['id']}.json", 'w'), indent=4)
# print("--- %s seconds ---" % (time.time() - start_time))
# print("Done")

# print("Upserting cands to database...")
# start_time = time.time()
# for resume in resumes:
#     desc = upsert_to_database("candidate", resume)
#     # json.dump({'id': resume['id'], 'description':desc}, open(f"../new_data/cands/{resume['id']}.json", 'w'), indent=4)
# print("--- %s seconds ---" % (time.time() - start_time))
# print("Done")

# time.sleep(5)

print("Getting candidate scores...")
cands_data = []
for i in range(1, 4):
    cands_data.append(json.load(open(f"../new_data/cands/{i}.json")))
print("Data Loaded")
jdk1 = json.load(open(f"../new_data/jdks/1.json"))
start_time = time.time()
results = get_candidate_scores(jdk1, cands_data)
print("--- %s seconds ---" % (time.time() - start_time))
print("JDK 1 Done")
# print(results)

jdk2 = json.load(open(f"../new_data/jdks/2.json"))
start_time = time.time()
results = get_candidate_scores(jdk2, cands_data)
print("--- %s seconds ---" % (time.time() - start_time))
print("JDK 2 Done")
# # print(results)
jdk3 = json.load(open(f"../new_data/jdks/3.json"))
start_time = time.time()
results = get_candidate_scores(jdk3, cands_data)
print("--- %s seconds ---" % (time.time() - start_time))
print("JDK 3 Done")
# print(results)
# print("--- %s seconds ---" % (time.time() - start_time))

# time.sleep(5)

# print("Getting job suggestions...")
# start_time = time.time()
# cand_data = json.load(open(f"../new_data/cands/1.json"))
# jobs_data = []
# for i in range(1, 4):
#     jobs_data.append(json.load(open(f"../new_data/jdks/{i}.json")))
# # print(jobs_data)
# result = get_job_suggestions(cand_data, jobs_data)
# print("Candidate 1 Done")
# # print(result)

# cand_data = json.load(open(f"../new_data/cands/2.json"))
# result = get_job_suggestions(cand_data, jobs_data)
# print("Candidate 2 Done")
# # print(result)

# cand_data = json.load(open(f"../new_data/cands/3.json"))
# result = get_job_suggestions(cand_data, jobs_data)
# print("Candidate 3 Done")
# # print(result)
# print("--- %s seconds ---" % (time.time() - start_time))

# time.sleep(5)

# print("Deleting all data...")
# start_time = time.time()

# jdk_id = '3'
# upsert_model = Upsert_model({'id': jdk_id})
# upsert_model.delete_jdk()


# for i in range(1, 4):
#     cand_id = str(i)
#     can_projects = json.load(open(f"../new_data/cands/{cand_id}.json"))['projects']
#     project_titles = [project['title'] for project in can_projects]

#     upsert_model = Upsert_model({'id': cand_id, 'projects': project_titles})
#     upsert_model.delete_candidate()

print("Total Time Taken: --- %s seconds ---" % (time.time() - abs_start_time))