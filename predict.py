from model import *
from preprocess import *
from data import *
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


model.load_weights('./model_weights/model_new2.h5')

def predict_caption(image):
    
    start_se = "<start>"
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in start_se.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')
        
        ypred = model.predict([image,sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        start_se += (' ' + word)
        
        if word == "<end>":
            break
    
    final_caption = start_se.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

plt.style.use("ggplot")
for i in range(15):
    idx = np.random.randint(0,1000)
    all_img_names = list(encoding_test.keys())
    img_name = all_img_names[idx]
  
    image_full = encoding_test[img_name].reshape((1,2048))
    
    i = plt.imread(IMG_path+img_name+".jpg")
    caption = predict_caption(image_full)
    
    plt.title(caption)
    plt.imshow(y)
    plt.axis("off")
    plt.savefig("image_captioned_"+str(i*3)+'.jpg')
 

    plt.show()
