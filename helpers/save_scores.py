import pandas as pd

def __flatten_dict(candidate_scores):
    """
    Flatten the dictionary to a list of tuples
    """

    flat_dict = {
        "id" : [],
        # "name" : [],
        "final_score" : [],
        "project_1_name" : [],
        "project_1_score" : [],
        "project_2_name" : [],
        "project_2_score" : [],
        "project_3_name" : [],
        "project_3_score" : [],
        "project_4_name" : [],
        "project_4_score" : [],
        "project_5_name" : [],
        "project_5_score" : [],
    }

    for candidate in candidate_scores:
        flat_dict['id'].append(candidate['id'])
        # flat_dict['name'].append(candidate['name'])
        flat_dict['final_score'].append(candidate['final_score'])

        project_scores = candidate['project_scores']
        for i in range(5):
            if i < len(candidate['project_scores']):
                flat_dict[f"project_{i+1}_name"].append(project_scores[i]['name'])
                flat_dict[f"project_{i+1}_score"].append(project_scores[i]['score'])
            else:
                flat_dict[f"project_{i+1}_name"].append('')
                flat_dict[f"project_{i+1}_score"].append('')

    return flat_dict


def save_to_excel(jdk_id, candidate_scores):
    """
    Save the dictionary to an excel file
    """

    excel_path = f"./results/jdk_{jdk_id}_resume_score.xlsx"

    flat_dict = __flatten_dict(candidate_scores)

    df = pd.DataFrame(flat_dict)
    df.to_excel(excel_path, index=False)

    return 