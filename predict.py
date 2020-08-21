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

from keras.applications.vgg16 import decode_predictions

from tensorflow.python.keras import layers

from tensorflow.python.keras import models

import tensorflow as tf

from tensorflow.keras.callbacks
import TensorBoard, EarlyStopping



def to_rgb(im):
    w, h = im.shape
   
 ret = np.empty((w, h, 3), dtype=np.uint8)
   
 ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] =  im
   
 return ret

graph=None
tmodel=None
lmodel=None
model=None

class 
 DeepModel:
    
    def __init__(self):
        np.random.seed(42)
        tf.set_random_seed(42)
        global tmodel
        global lmodel
        tmodel = VGG16(weights='imagenet', include_top=False)
        lmodel=models.Sequential()
        
lmodel.add(layers.Dense(50,activation="relu",input_shape=(4608,)))
        
lmodel.add(layers.Dense(30,activation="relu"))
        lmodel.add(layers.Dense(30,activation="relu"))
   
lmodel.add(layers.Dense(30,activation="relu"))
        lmodel.add(layers.Dense(18,activation="softmax"))
      
lmodel.summary()
       
lmodel.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
        
      
  global graph
        graph = tf.get_default_graph()

    def predict(self,data_file):
        global graph
        global tmodel
        global lmodel
        with graph.as_default():
            Qry = Image.open(os.path.join('C:/Users/shail',data_file))
            Qry = np.array(Qry.resize((110,110)))
            Qry = to_rgb(Qry)
            Qry = Qry.reshape([-1,110,110,3])
            features_train=tmodel.predict([Qry])
            lmodel.load_weights('my_checkpoint')
           
  return lmodel.predict_classes([features_train.flatten().reshape((1, 4608))])




# In[ ]:



  

from flask import Flask, render_template, request

from werkzeug import secure_filename
app = Flask(__name__)

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
  
   if request.method == 'POST':
     
    f = request.files['file']
        
 f.save(secure_filename("qry.png"))
        file_name='qry.png'
        return str(model.predict(file_name)[0])
    
    
if __name__ == '__main__':
    global model
    model=DeepModel()
    app.run(debug = False)



