import pdfplumber
from flask import Flask, jsonify, request
from flask import Flask
from flask_cors import CORS, cross_origin
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

app = Flask(__name__)


@app.route('/pdf', methods=['POST'])
def get_text():
    if 'file' not in request.files:
        return 'No file part in the request', 400

    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400
    extractedInformation = ""
    if file and file.filename.endswith('.pdf'):
        num_pages_to_extract = 1

        text = ''
        with pdfplumber.open(file) as pdf:
            for page_num in range(min(num_pages_to_extract, len(pdf.pages))):
                page = pdf.pages[page_num]
                page = page.dedupe_chars(tolerance=1)
                page_text = page.extract_text()
                text += page_text + '\n'
                extractedInformation = text
    elif file.filename.endswith(('.jpg', '.jpeg', '.png', '.PNG')):
        print("Image received")
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        img_pillow = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        extractedInformation = pytesseract.image_to_string(img_pillow)
    else:
        print('Invalid file format. Please upload a PDF file.')
    return {'text': extractedInformation}


app.run()
