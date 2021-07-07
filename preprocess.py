import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import pickle
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model,Sequential,load_model
from keras.layers import *
import re
import json
import collections
import time
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image

def load_doc(path):
    
    with open(path, encoding='utf8') as file:
        text = file.read()
    return text
  
text = load_doc("/content/all_captions/Flickr8k.token.txt")

captions = text.split('\n')[:-1] #last line is empty

captions[0]

cap = captions[0].split('\t')
img_id = cap[0].split('.')[0]
caption = cap[1]

def get_img_caption_mapping(text):
    description = {}
    for ix in range(len(captions)):
        tex = captions[ix].split('\t')
       
        img_id = tex[0].split('.')[0]
        
        cap = tex[1]
        
        if description.get(img_id) is None:
            description[img_id] = []
        description[img_id].append(cap)
        
    return description

descriptions = get_img_caption_mapping(captions)

IMG_path = "/content/all_images/Flicker8k_Dataset/"
img = image.load_img(IMG_path+"1000268201_693b08cb0e.jpg",target_size=(299, 299, 3))
img = image.img_to_array(img)/255.
plt.imshow(img)

def cleaned_captions(caption):

    caption = caption.lower()
    caption = re.sub("[^a-z]+"," ", caption)
    caption = [x for x in caption.split() if len(x)>1]
    caption = " ".join(caption)
    return sentence

for img_id,caption_list in descriptions.items():
    
    for i in range(len(caption_list)):
        
        caption_list[i] = clean_captions(caption_list[i])

with open("description.txt",'w',encoding='utf8') as f:
    f.write(str(descriptions))

description = None
with open("description.txt",'r') as f:
    description = f.read()
    
json_acceptable_string = description.replace("'","\"")
description = json.loads(json_acceptable_string)

vocab = set()
total_words = []

for key in description.keys():
    [vocab.update(sentence.split()) for sentence in description[key]]
    [total_words.append(i) for des in description[key] for i in des.split()]

print(len(total_words),len(vocab))

counter = collections.Counter(total_words)
freq_count = dict(counter)

print(freq_count['the'])

sorted_freq_cnt = sorted(freq_count.items(),reverse=True,key = lambda x:x[1])
threshold = 10
sorted_freq_cnt = [x for x in sorted_freq_cnt if x[1]>threshold]
total_words = [x[0] for x in sorted_freq_cnt]

print(len(total_words)) 

train = load_doc("/content/all_captions/Flickr_8k.trainImages.txt")
test = load_doc("/content/all_captions/Flickr_8k.testImages.txt")
