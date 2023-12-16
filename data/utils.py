import json
import random

with open('data/resumes.json', 'r') as file:
    data = json.load(file)

# print(data)

n = len(data)

for i in range(n):
    m = len(data[i]['projects'])
    for j in range(m):
        data[i]['projects'][j]['experience'] = random.randint(1, 12)
        
with open('data/resumes_2.json', 'w') as file:
    json.dump(data, file, indent=4)