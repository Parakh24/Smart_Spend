from flask import Flask, render_template, request 
import joblib 
import numpy as np 
from joblib import load 

deploy = Flask(__name__) 


#load the model 
model = joblib.load(open('savemodel.joblib' , 'rb')) 


@deploy.route('/') 

def home():
    result = '' 
    return render_template('index.html' , **locals()) 


@deploy.route('/predict' , methods = ['POST' , 'GET']) 

def predict(): 


    V1 = int(request.form[V1])
    V2 = int(request.form[V2])
    V3 = int(request.form[V3]) 
    V4 = int(request.form[V4])  
    input_data =  np.array([[V1 , V2 , V3 , V4]]) 
    result = int(model.predict(input_data)[0]) 
    return render_template('index.html' , **locals()) 

if __name__ == '__main__': 
    deploy.run(debug = True) 


