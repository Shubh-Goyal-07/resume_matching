# Resume Matching

## Adding and Deleting Data

First, define an instance of the Manager_model() class which is present in the `/main/dataManager.py` file.

### For Candidates

- To add the candidate data, call the **add_candidate()** function using the created instance.

  - Parameters: data (type: dictionary)
  - Returns: string
  - Structure of the data:  
```javascript
            {
                 "id": string/integer,  
                 "projects": array -  
                       [  
                            {  
                                 "title":  
                                 "description":  
                                 "skills":  
                                 "endDate":  
                                "startDate":  
                            },  
                            {...  
                            }  
                            ...  
                       ]  
            }
```

- To delete the candidate data, call the **delete_candidate()** function using the created instance.

  - Parameters: data (type: dictionary)
  - Returns: None
  - Structure of the data:  
```javascript
            {  
                 "id": string/integer,  
                 "projects": array - [project1_title, project2_title,...]
            }
```

### For JDK

- To add a new JDK, call the **add_jdk()** function using the created instance.

  - Parameters: data (type: dictionary)
  - Returns: string
  - Structure of the data:  
```javascript
            {  
                 "id": string/integer,
                 "title": string,
                 "description": string,
                 "skills": array - [skill1, skill2...]
            }
```

- To delete a JDK, call the **delete_jdk()** function using the created instance.

  - Parameters: data (type: dictionary)
  - Returns: None
  - Structure of the data:  
```javascript
            {  
                 "id": string/integer
            }
```

## Scoring

### Candidate Scoring for a specific JDK

Call the **get_candidate_scores()** which is present in the `/main/hrAssistant.py` file.
  - Parameters: jdk_info (type: dictionary), candidates_info (type: array)
  - Returns: array
  - Structure of jdk_info:
```javascript
            {  
                  "id": string/integer,
                  "description": string,
                  "softSkills": array
            }
```
  - Structure of candidates_info:
```javascript
            [
                 {
                      "id": string/integer,
                      "description": string,
                      "compRecruitScreeningAnswers": json,
                      "compRecruitQuestionnaireAnswers": json
                 },
                 {...  
                 }  
                 ...  
            ]
```
  - Structure of returned array:
```javascript
            [
                 {
                      "id": string/integer,
                      "final_score": float,
                      "reason": string,
                      "personality_score": float,
                      "personality_reason": string
                 },
                 {...  
                 }  
                 ...  
            ]
```

### Job Scoring for a specific candidate

Call the **get_job_suggestions()** which is present in the `/main/candAssistant.py` file.
  - Parameters: candidate_info (type: dictionary), jdks_info (type: array)
  - Returns: array
  - Structure of candidate_info:
```javascript
            {  
                  "id": string/integer,
                  "description": string
            }
```
  - Structure of jdks_info:
```javascript
            [
                 {
                      "id": string/integer,
                      "description": string
                 },
                 {...  
                 }  
                 ...  
            ]
```
  - Structure of returned array:
```javascript
            [
                 {
                      "id": string/integer,
                      "score": float,
                      "reason": string
                 },
                 {...  
                 }  
                 ...  
            ]
```