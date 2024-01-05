from upsert import upsert_to_database
from hrassistant import score_candidates
from candassistant import get_job_suggestions
import json
import time

jdks = json.load(open('../data/jdks.json'))
resumes = json.load(open('../data/resumes.json'))

abs_start_time = time.time()

# print("Upserting jdks to database...")
# start_time = time.time()
# for jdk in jdks:
#     desc = upsert_to_database("jdk", jdk)
#     json.dump({'id': jdk['id'], 'description':desc}, open(f"../new_data/jdks/{jdk['id']}.json", 'w'), indent=4)
# print("--- %s seconds ---" % (time.time() - start_time))
# print("Done")

print("Upserting cands to database...")
start_time = time.time()
for resume in resumes:
    desc = upsert_to_database("candidate", resume, save_gen_desc_only=False)
    json.dump({'id': resume['id'], 'description':desc['final_description'], 'personality_score':desc['score']}, open(f"../new_data/cands/{resume['id']}.json", 'w'), indent=4)
print("--- %s seconds ---" % (time.time() - start_time))
print("Done")

# time.sleep(5)

# print("Getting candidate scores...")
# start_time = time.time()
# cands_data = []
# for i in range(1, 4):
#     cands_data.append(json.load(open(f"../new_data/cands/{i}.json")))
# print("Data Loaded")
# jdk1 = json.load(open(f"../new_data/jdks/1.json"))
# results = score_candidates(jdk1, cands_data)
# print("JDK 1 Done")
# print(results)
# jdk2 = json.load(open(f"../new_data/jdks/2.json"))
# results = score_candidates(jdk2, cands_data)
# print("JDK 2 Done")
# print(results)
# jdk3 = json.load(open(f"../new_data/jdks/3.json"))
# results = score_candidates(jdk3, cands_data)
# print("JDK 3 Done")
# print(results)
# print("--- %s seconds ---" % (time.time() - start_time))

# time.sleep(5)

# print("Getting job suggestions...")
# start_time = time.time()
# result = get_job_suggestions(1)
# print("Candidate 1 Done")
# print(result)
# result = get_job_suggestions(2)
# print("Candidate 2 Done")
# print(result)
# result = get_job_suggestions(3)
# print("Candidate 3 Done")
# print(result)
# print("--- %s seconds ---" % (time.time() - start_time))

print("Total Time Taken: --- %s seconds ---" % (time.time() - abs_start_time))