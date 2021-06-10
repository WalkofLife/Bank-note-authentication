from flask import Flask, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome to Bank Note Authentication"

@app.route('/predict')
def predict():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    kurtosis = request.args.get('kurtosis')
    entropy = request.args.get('entropy')

    prediction = classifier.predict([[variance, skewness, kurtosis, entropy]])

    return "The predicted Classification is {}".format(prediction)


@app.route('/predict_file', methods = ['POST'])
def predict_file():
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    return "Prediction value for the csv is {}".format(str(list(prediction)))
  






if __name__ == '__main__':
    app.run()