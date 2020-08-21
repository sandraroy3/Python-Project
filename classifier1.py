#!/usr/bin/env python

# coding: utf-8


# In[1]:


import os

from PIL import Image

import numpy as np

import os

import cv2

from matplotlib import pyplot as plt

from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input

import numpy as np

from keras.applications.vgg16 
import decode_predictions


model = VGG16(weights='imagenet', include_top=False)


X = []

Y = []


base_path='D:/FaceRecognition/datastet2'

source_path=base_path

for child in os.listdir(source_path):
 
   sub_path = os.path.join(source_path, child)
    
   bsub_path = os.path.join(base_path, child)
  
   if os.path.isdir(sub_path):
      
    for data_file in os.listdir(sub_path):
          
     Qry = Image.open(os.path.join(sub_path, data_file))
            
     Qry = np.array(Qry.resize((224,224)))
            
     Qry = Qry.reshape([-1,224,224,3])
          
     features_train=model.predict([Qry])
            
     X.append(features_train.flatten())
           
     Y.append(child)

print(X)

print(Y)




# In[2]
:


from sklearn.preprocessing import LabelBinarizer

labelBinarizer = LabelBinarizer()

y = labelBinarizer.fit_transform(Y)

print(y)
       

 


# In[3]:



from sklearn.model_selection import train_test_split
X_train,
X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42)


import tensorflow as tf


from tensorflow.python.keras import layers

from tensorflow.python.keras import models


import random


random.seed(42)

np.random.seed(42)

tf.set_random_seed(42
)


dnnModel=models.Sequential()


dnnModel.add(layers.Dense(50,activation="relu",input_shape=(25088,)))

dnnModel.add(layers.Dense(30,activation="relu"))



dnnModel.add(layers.Dense(30,activation="relu"))



dnnModel.add(layers.Dense(30,activation="relu"))



dnnModel.add(layers.Dense(21,activation="softmax"))


dnnModel.summary()



dnnModel.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


tbCallBack=tf.keras.callbacks.TensorBoard(log_dir='./Graph',histogram_freq=0,write_graph=True,write_images=True)


dnnModel.fit(X_train,y_train,epochs=50,batch_size=64,callbacks=[tbCallBack])

testloss, testAccuracy=dnnModel.evaluate(X_test,y_test)

dnnModel.save_weight('my_checkpoint')

print(testAccuracy)




