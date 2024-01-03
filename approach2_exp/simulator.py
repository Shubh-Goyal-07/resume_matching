from embd_upsert import upsert_to_database
from hrassistant import score_candidates
from candassistant import get_job_suggestions
import json
import time

jdks = json.load(open('../data/jdks.json'))
resumes = json.load(open('../data/resumes.json'))

abs_start_time = time.time()

# print("Upserting to database...")
# start_time = time.time()
# for jdk in jdks:
#     upsert_to_database("jdk", jdk)
# print("--- %s seconds ---" % (time.time() - start_time))
# print("Done")

# print("Upserting to database...")
# start_time = time.time()
# print("Resume 1")
# resume1 = resumes[0]
# upsert_to_database("candidate", resume1, save_gen_desc_only=True)
# print("Resume 2")
# resume2 = resumes[1]
# upsert_to_database("candidate", resume2)
# print("Resume 3")
# resume3 = resumes[2]
# upsert_to_database("candidate", resume3, save_gen_desc_only=False)
# for resume in resumes:
    # upsert_to_database("candidate", resume, save_gen_desc_only=True)
# print("--- %s seconds ---" % (time.time() - start_time))
# print("Done")

# time.sleep(5)

# print("Getting candidate scores...")
# start_time = time.time()
# score_candidates(1, [3, 2])
# print("JDK 1 Done")
# score_candidates(2, [2, 3])
# print("JDK 2 Done")
# score_candidates(3, [2, 3])
# print("JDK 3 Done")
# print("--- %s seconds ---" % (time.time() - start_time))

# time.sleep(5)

print("Getting job suggestions...")
start_time = time.time()
get_job_suggestions(1)
print("Candidate 1 Done")
get_job_suggestions(2)
print("Candidate 2 Done")
get_job_suggestions(3)
print("Candidate 3 Done")
print("--- %s seconds ---" % (time.time() - start_time))

print("Total Time Taken: --- %s seconds ---" % (time.time() - abs_start_time))