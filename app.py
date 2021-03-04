from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import logging

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

logging.basicConfig(level=logging.INFO)
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
EMOTIONS_MODEL_PATH = 'models/emotion_model.hdf5'

# Load your trained model
model = load_model(EMOTIONS_MODEL_PATH)
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

# ---start--- # First time uncomment this

# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
# model.save(MODEL_PATH)
# print('Model loaded. Check http://127.0.0.1:5000/')

#---end---
emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(48, 48),color_mode='grayscale')

    # Preprocessing the image
    img = image.img_to_array(img)  # (48,48,1)
    # x = np.true_divide(x, 255)
    img = np.expand_dims(img, axis=0) #(1,48,48,1)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')
    # preds = model.predict(x)
    predicted_class = np.argmax(model.predict(img))
    label_map = dict((v,k) for k,v in emotion_dict.items())
    preds = label_map[predicted_class]

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        logging.info(f'image file path {file_path}')
        f.save(file_path)
        result = model_predict(file_path,model)
        return result
        # Make prediction
        # preds = model_predict(file_path, model)

        # # Process your result for human
        # # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        # return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

