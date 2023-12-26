from embd_upsert import upsert_to_database
import json
import time

jdks = json.load(open('../data/jdks.json'))
resumes = json.load(open('../data/resumes.json'))

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