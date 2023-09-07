from __future__ import print_function
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import pickle
import requests
from flask_socketio import SocketIO, emit
import warnings
warnings.filterwarnings('ignore')

import sys
from json import *
from flask_cors import CORS, cross_origin

app =   Flask(__name__)
CORS(app)
socketio = SocketIO(app)
crop_recommendation_model_path = './RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))
  
@app.route('/crop', methods = ['GET'])
def ReturnJSON():
    N = request.args.get('N')
    P = request.args.get('P')
    K = request.args.get('K')
    ph = request.args.get('pH')
    humidity = request.args.get('humidity')
    avg_rainfall = request.args.get('avg_rainfall')
    temprature = request.args.get('temprature')

    data = np.array([[N, P, K, temprature, humidity, ph, avg_rainfall]])
    my_prediction = crop_recommendation_model.predict(data)
    final_prediction = my_prediction[0]

    return jsonify({"crop": final_prediction})
  
if __name__=='__main__':
     socketio.run(app,  allow_unsafe_werkzeug=True)
