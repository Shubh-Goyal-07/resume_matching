from google.cloud import translate_v2 as translate

import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
api_key_string = os.environ['GOOGLE_TRANSLATE_API_KEY']
project_id = os.environ['GOOGLE_PROJECT_ID']

def translate_text(text, target='en'):
    # translate_client = translate.Client()
    translate_client = translate.Client(client_options={"api_key": api_key_string, "project_id": project_id})
    
    result = translate_client.translate(text, target_language=target)

    print("Text: {}".format(result["input"]))
    print("Translation: {}".format(result["translatedText"]))
    print("Detected source language: {}".format(result["detectedSourceLanguage"]))

    return result


translate_text('Hola Mundo!')