from embd_upsert import upsert_to_database
from embed_query import get_candidate_score
import json
import time

jdks = json.load(open('../data/jdks.json'))
resumes = json.load(open('../data/resumes.json'))

abs_start_time = time.time()

print("Upserting to database...")
start_time = time.time()
for jdk in jdks:
    upsert_to_database("jdk", jdk)
print("--- %s seconds ---" % (time.time() - start_time))
print("Done")

print("Upserting to database...")
start_time = time.time()
for resume in resumes:
    upsert_to_database("candidate", resume)
print("--- %s seconds ---" % (time.time() - start_time))
print("Done")

time.sleep(5)

print("Getting candidate scores...")
start_time = time.time()
get_candidate_score(1, [1, 3, 2])
print("JDK 1 Done")
get_candidate_score(2, [1, 2, 3])
print("JDK 2 Done")
get_candidate_score(3, [1, 2, 3])
print("JDK 3 Done")
print("--- %s seconds ---" % (time.time() - start_time))

print("Total Time Taken: --- %s seconds ---" % (time.time() - abs_start_time))