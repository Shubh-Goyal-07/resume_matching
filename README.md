### **upsert.py - upsert\_to\_database(category, data)**

_**Parameters:**_

1.  _**category:**_ string - "jdk" or "candidate"
2.  _**data:**_ dictionary

         if “jdk” -

            {

                 "id": string/integer,    
                 "title": string,    
                 "description": string,    
                 "skills": array    
            }

         if “candidate” -  
            {  
                  "id": string/integer  
                  "projects": array -  
                       \[  
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
                       \]  
            }

_**Returns:**_ string
  
---
  

### **hrassistant.py - score\_candidates(jdk\_info, candidates\_info)**

_**Parameters:**_

 1. _**jdk\_info:**_ dictionary

            {

                 “id”: string/integer

                 “description”: string

                 “softSkills”: array

            }

    2. _**candidates\_info:**_ array

            \[

                 {

                      “id”: string/integer

                      “description”: string

                      “compRecruitScreeningAnswers”:

                      “compRecruitQuestionnaireAnswers”:

                 },

                 {…

                 },

                 ….

            \]

_**Returns:**_ array

        \[

             {

                  “id”:

                  “final\_score”:

                  “reason”:

                  “personality\_score”:

                  “personality\_reason”:

             }

        \]

  ---

  ### **candassistant.py - get\_job\_suggestions**
