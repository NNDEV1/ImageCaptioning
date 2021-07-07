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
from data import *

input_img = Input(shape=(2048, ))
image_model1 = Dropout(0.3)(input_img)
image_model2 = Dense(256, activation='relu')(image_model1)

input_cap = Input(shape=(max_len, ))
caption_model1 = Embedding(vocab_size, 200, mask_zero=True)(input_cap)
caption_model2 = Dropout(0.3)(caption_model1)
caption_model3 = LSTM(256)(caption_model2)

decoder = add([image_model2, caption_model3])
decoder_2 = Dense(256, activation='relu')(decoder)
output = Dense(vocab_size, activation='relu')(decoder_2)

model = Model(inputs=[input_img, input_cap], outputs=output)

print(model.summary())

model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

epochs = 10
batch_size = 3
steps = len(train_description)//batch_size
generator = data_generator(train_description,encoding_train,word_index,max_len,batch_size)

def train():
    for i in range(epochs):
       
        model.fit_generator(generator,epochs=1,steps_per_epoch=steps,verbose=1)
        if i%2==0:
            model.save("/content/weights/final_model"+str(i*2)+'.h5')

train()
