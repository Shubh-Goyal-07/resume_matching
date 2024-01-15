# resume_matching

## **upsert.py - upsert_to_database(category, data)**

***Parameters:***


> 1. category : string - "jdk" or "candidate"
> 2. data : dictionary
            if "jdk" - 
            {
              "id": string/integer
              "title": string
              "description": string
              "skills": array
            }
            if "candidate" -
            {
              "id": string/integer
              "projects": array -
                    [{"title":
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

***Returns:***
string - concise and precise desription of the jdk or candidate
