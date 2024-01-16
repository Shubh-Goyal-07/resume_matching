# Resume Matching

## Adding and Deleting Data

First, define an instance of the Manager_model() class which is present in the `/main/dataManager.py` file.

### For Candidates

To add the candidate data, call the **add_candidate()** function using the created instance.

- Parameters: data (type: dictionary)
- Returns: string
- Structure of the data:  
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
