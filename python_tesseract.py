
import pytesseract
import re
import shutil
import os
import random
from IPython.display import display
try:
 from PIL import Image
except ImportError:
 import Image
import cv2
import numpy as np
from IPython.display import display, Image as IPImage
import requests
# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)

#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

# Apply adaptive thresholding
url = 'https://www.researchgate.net/publication/337702424/figure/fig3/AS:831888872730624@1575349171831/Students-Paragraph-Writing.png'

# Download the image using requests
response = requests.get(url)
img_array = np.asarray(bytearray(response.content), dtype=np.uint8)

# Read the image using cv2
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
# img = cv2.imread('uber_test4')

gray = get_grayscale(img)
# noise=remove_noise(gray) dont do it!
# thresh = thresholding(gray) nahh output is not good!
# opening = opening(gray)  dont use!
# canny = canny(gray)   dont use!

# inverted_binary_image = cv2.bitwise_not(gray)
# Image.fromarray(inverted_binary_image).show()

#inverting image color not helping

img_pillow = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# image_path_in_colab="/content/ocr_test4.PNG"
extractedInformation = pytesseract.image_to_string(img_pillow)
print(extractedInformation)

import spacy
nlp = spacy.load("en_core_web_sm")

doc=nlp(extractedInformation)
for ent in doc.ents:
  if ent.label_== "DATE":
    temp=ent.text
  print(ent.text, "|", ent.label_, "|", spacy.explain(ent.label_))

"""# **Using Bert Base NER**"""


from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

ner_results = nlp(extractedInformation)
print(ner_results)

"""# **Using Bert-large-NER**"""

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
ner_results = nlp(extractedInformation)
print(ner_results)

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

# Extract addresses
addresses = extract_address(ner_results)

# Print the addresses
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

  # Print the result
  # print(result_string)
  return result_string

correctTokens(addresses[9])

corrected_addresses = []

# Iterate through the original addresses and store the corrected ones in the new list
for address in addresses:
    corrected_addresses.append(correctTokens(address))

# Print the corrected addresses
for address in corrected_addresses:
    print(address)

indices_of_india = [i for i, x in enumerate(corrected_addresses) if "India" in x]

# Check if at least two occurrences of "India" are found
if len(indices_of_india) >= 2:
    # Extract the substring from the beginning to the first occurrence of "India"
    substring_before_first_india = corrected_addresses[:indices_of_india[0] + 1]

    # Extract the substring from the first occurrence of "India" to the second occurrence of "India"
    substring_between_indias = corrected_addresses[indices_of_india[0] + 1:indices_of_india[1] + 1]

first_address = '\n'.join(substring_before_first_india)
second_address= '\n'.join(substring_between_indias)

# Print the combined string
print(first_address)
print(second_address)

