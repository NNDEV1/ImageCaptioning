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
from preprocess import *


model = ResNet50(weights='imagenet', input_shape=(299, 299, 3))
model.summary()

model_new = Model(model.input, model.layers[-2].output)

def preprocessing(imag):
    img = image.load_img(imag, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    return img

def encode(img):
    img = preprocessing(img)
    feature_vector = model_new.predict(img)
    feature_vector = feature_vector.reshape((2048, ))

    return feature_vector

print(encode(IMG_path+"1000268201_693b08cb0e.jpg").shape)

encoding_train = {}

for i,img_id in enumerate(train):
    
    path = IMG_path+img_id+".jpg"
    encoding_train[img_id] = encode(path)
    
    
encoding_test = {}

for i,img_id in enumerate(test):
    
    img_path = IMG_path+img_id+".jpg"
    encoding_test[img_id] = encode_img(img_path)
    
with open("/content/pickle/training_encoded.pkl",'wb') as f:
    pickle.dump(encoding_train,f)
with open("/content/pickle/testing_encoded.pkl",'wb') as f:
    pickle.dump(encoding_test,f)
    
    
word_index = {}
index_word = {}

for idx,word in enumerate(total_words):
    word_index[word] = idx + 1
    index_word[idx+1] = word

print(len(index_word))

word_index['<start>']= 1846
word_index['<end>'] = 1847

index_word[1846] = '<start>'
index_word[1847] = '<end>'

vocab_size = len(word_index)+1
print(vocab_size)

max_len = 0
for key in train_description.keys():
    
     for cap in train_description[key]:
            max_len = max(max_len,len(cap.split()))


def data_generator(captions,encoding_image,word_index,max_len,batch_size):
    xa, xb, y = [],[],[]
    
    n = 0
    while True:
        for key, cap_list in captions.items():
            n += 1
            
            image = encoding_image[key]
            for cap in cap_list:
                
                seq = [word_index[word] for word in cap.split() if word in word_index]
                for i in range(1,len(seq)):
                    xi = seq[0:i]
                    yi = seq[i]
                    
                    xi = pad_sequences([xi],maxlen=max_len,value=0,padding='post')[0]
                    yi = to_categorical([yi],num_classes=vocab_size)[0]
                    
                    xa.append(image)
                    xb.append(xi)
                    y.append(yi)
                    
                if (n == batch_size):
                    yield [[np.array(xa),np.array(xb)],np.array(y)]
                    xa, xb, y = [], [], []
                    n = 0

a = data_generator(train_description,encoding_train,word_index,max_len,batch_size=1)


f= open("glove.6B.200d.txt",'r',encoding='utf8')

embedding_index = {}
for line in f:
    values = line.split()
    word = values[0]
    vector = np.array(values[1:],dtype='float32')
    
    embedding_index[word] = vector

f.close()

def get_matrix(): 
    emb_dim = 200
    matrix = np.zeros((vocab_size,emb_dim))
    
    for word,idx in word_index.items():
        vector = embedding_index.get(word)
        
        if vector is not None:
            matrix[idx] = vector
            
    return matrix

embedding_matrix = get_matrix()
