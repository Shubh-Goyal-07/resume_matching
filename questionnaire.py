from openai import OpenAI
import os

class questionnaire_scoring():
    def __init__(self):
        self.prompt = """You are an agent that judges an applicant's personality and his/her willingness to go to Japan to work for the company. 
        
        You have to judge the willingness of the applicant to go to Japan based on his/her answers to the following set of questions:
        1. The reason why you want to come to Japan
        2. The career plan you want
        3. In which country do you want to work after graduation?
        4. Are you likely to be adaptable to other cultures?
        5. Instead of English-speaking countries like the U.S., the U.K. and Singapore, why are you interested in working in Japan?
        6. What are your expectations from the company?
        7. What would you like to accomplish during the internship with the company?

        In addition to the willingness, you also have to judge the personality of the applicant based on his/her answers to the following set of questions:
        1. Your strengths and characteristics
        2. Weaknesses or areas where they would like to improve
        3. Steps they are taking or plan to take to address these areas
        4. Example of a challenge or setback you have faced and how they overcame it
        5. Lessons learned from that experience
        6. Do you have any unique background that differentiate you from others?
        7. Do you have any specialty?

        You will be given the answers to all the questions in a JSON format. You have to give a single score based on the applicant's personality and his/her willingness to go to Japan.

        You have to return the output in the following JSON format:
        {
            Score: <GIVE A SCORE OUT OF 5 HERE>,
        }
        """
        # Reasoning: <GIVE A BRIEF REASONING FOR THE SCORE. BE SUCCINT.>

    def __init_client(self, OPENAI_API_KEY):
        self.client = OpenAI(
            api_key = OPENAI_API_KEY
        )

        return

    def __calculate_score(self, answers):
        self.response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an agent that judges an applicant's personality and his/her willingness to go to Japan to work for the company."},
                {"role": "user", "content": self.prompt + '\n' + "Here are the answers:\n" + str(answers)},
            ]
        )

        return self.response.choices[0].message.content