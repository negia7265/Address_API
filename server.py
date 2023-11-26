import pytesseract
import re
import shutil
import os
import random
import json
from IPython.display import display
try:
    from PIL import Image
except ImportError:
    import Image
import cv2
import numpy as np
from IPython.display import display, Image as IPImage
from flask import Flask, jsonify, request
from flask import Flask
from flask_cors import CORS, cross_origin
from transformers import AutoTokenizer, TFAutoModelForTokenClassification
from transformers import pipeline

app = Flask(__name__)


@app.route('/process_image', methods=['POST'])
def unique_process_image_endpoint9():
    data_received = request.get_json()
    image_data = data_received.get('name')
    return ({'my_name_issss_slim_shady': image_data, 'last_name_issss': image_data})

@app.route('/test_route',methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    print(file.content_type)
    kind=""
    if(file.content_type=="application/pdf"):
        kind="PDF"
    elif(file.content_type=="image/png"):
        kind="IMG"
    else:
        kind="INVALID"
    return {'status':'File uploaded successfully','type_of_file':kind}


@app.route('/upload', methods=['POST'])
def upload_file32():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    print(type(file))
    #converting
    image_data = file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    

    #converting image to text
    img_pillow = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    extractedInformation = pytesseract.image_to_string(img_pillow)
    
    #might well have to change the model

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    # model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    model=TFAutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    ner_results = nlp(extractedInformation)

    
    def extract_address(ner_result):
        addresses = []
        current_address = []
        for token in ner_result:
            if token['entity'].startswith('B-LOC'):
                if current_address:
                    addresses.append(" ".join(current_address))
                    current_address = [token['word']]
                else:
                    current_address.append(token['word'])
            elif token['entity'].startswith('I-LOC'):
                current_address.append(token['word'])

        if current_address:
            addresses.append(" ".join(current_address))

        return addresses


    addresses = extract_address(ner_results)
    for address in addresses:
        print(address)
    
    def correctTokens(address):
        split_address=address.split()
        tokens = split_address
        # Combine '##' tokens with the preceding non-'##' element
        combined_tokens = []
        current_token = None

        for token in tokens:
            if token.startswith('##'):
                if current_token is not None:
                    current_token += token[2:]  # Concatenate with the non-'##' part
            else:
                if current_token is not None:
                    combined_tokens.append(current_token)
                current_token = token

        # Add the last token if it exists
        if current_token is not None:
            combined_tokens.append(current_token)

        # Replace the first non-'##' token with the combined tokens
        if combined_tokens:
            tokens[0] = combined_tokens[0]

        # Remove '##' tokens from the list
        tokens = [token for token in combined_tokens if not token.startswith('##')]

        # Print the result
        # print(tokens)
        result_string = ' '.join(tokens)
        return result_string

    corrected_addresses = []


    #removing the # symbol and appending with the former non # token


    for address in addresses:
        corrected_addresses.append(correctTokens(address))
    

    cities=["Kolkata","Bengaluru","Dehradun","Chennai","Mumbai","Alwar","Jaipur"]
    states=["Rajasthan","Karnataka","Uttarakhand"]

    def IndiaExist(temp):
        for i in range(len(temp)):
            if(temp[i]=="India"):
                return True
    def CityExist(city):
        for i in range(len(cities)):
            if(cities[i]==city):
                return True
    def StateExist(state):
        for i in range(len(states)):
            if(states[i]==state):
                return True

    size = 5
    my_list = [0] * size
    list_again=[]
    if(IndiaExist(corrected_addresses)):
        for i in range(len(corrected_addresses)):
            if(corrected_addresses[i]=="India"):
                list_again.append(i)
    else:
        for i in range(len(corrected_addresses)):
            if(CityExist(corrected_addresses[i])):
                list_again.append(i)
    


    indices_of_india = [i for i, x in enumerate(corrected_addresses) if "India" in x]
    print(list_again)
    # Check if at least two occurrences of "India" are found
    for address in corrected_addresses:
        print(address)
    substring_before_first_india=[]
    substring_between_indias=[]
    if len(list_again) >= 2:
        # Extract the substring from the beginning to the first occurrence of "India"
        substring_before_first_india = corrected_addresses[:list_again[0] + 1]

        # Extract the substring from the first occurrence of "India" to the second occurrence of "India"
        substring_between_indias = corrected_addresses[list_again[0] + 1:list_again[1] + 1]

    #removing white spaces

    cleaned_first_address = [word.strip() for word in substring_before_first_india if word.strip()]
    cleaned_second_address = [word.strip() for word in substring_between_indias if word.strip()]

    first_address = ', '.join(cleaned_first_address)
    second_address= ', '.join(cleaned_second_address)

    # Print the combined string

    print(first_address)
    print(second_address)
    return {'status':'File uploaded successfully','source':first_address,'destination':second_address}
app.run()
