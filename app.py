# -*- coding: utf-8 -*-
"""
Created by 5th year software engineering
"""
from __future__ import division, print_function
from crypt import methods
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
# from list import data
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from ListImages import name
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
# pip install cv2
# Model saved with Keras model.save()
MODEL_PATH ='model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    array = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    data=array/255
    print(data)

    # print(x)
    expand = np.expand_dims(data, axis=0)
    print(len(expand.shape))

    predict = model.predict(expand)
    index=np.argmax(predict, axis=1)
    print(index)
    maximum_values = f'{data[np.arange(data.shape[1]), index]}'
    value = f'{data[index]}'
    if index==0:
        index="Bacterial spot"
    elif index==1:
        index="Early blight"
    elif index==2:
        index="Late blight"
    elif index==3:
        index="Leaf Mold"
    elif index==4:
        index="Septoria leaf spot"
    elif index==5:
        index="Spider mites Two spotted spider mite"
    elif index==6:
        index="Target Spot"
    elif index==7:
        index="Tomato Yellow Leaf Curl Virus"
    elif index==8:
        index="Tomato mosaic virus"
    elif index == 9:
        index="Healthy"
        model_predict.x = index
    return index


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/about', methods=['GET'])
def about():
    # Main page
    return render_template('About.html')

@app.route('/view', methods=['GET'])
def view():
    # Main page
    data = 'uploads'
    files = []
    for r, d, f in os.walk(data):
        for file in f:
            files.append(os.path.join(r, file))
        return render_template('view.html',
                               len = len(files),
                               f = files)
@app.route('/links', methods=['GET'])
def links():
    # Main page
    return render_template('links.html')


@app.route('/profile', methods=['GET'])
def profile():
    # Main page
    return render_template('profile.html')


@app.route('/camera', methods=['GET', 'POST'])
def camera():
    return render_template('camera.html')
@app.route('/copy', methods=['GET', 'POST'])
def copy():
    return render_template('copy.html')

@app.route('/prediction', methods=['GET'])
def prediction():
    # Main page
    return render_template('prediction.html')

    # Main page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
            # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join( basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

            # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None

if __name__ == '__main__':
    app.run(port=5000,debug=True)
