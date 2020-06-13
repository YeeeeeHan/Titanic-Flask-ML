import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS, cross_origin

# from tensorflow import keras
# from keras import utils
# from keras.utils import to_categorical
# from keras.models import Sequential, model_from_json, load_model
# from keras.layers import Dense, Activation, Dropout, Input
# from keras.layers.normalization import BatchNormalization
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences

import pickle

from keras.models import load_model


app = Flask(__name__)
api = Api(app)
CORS(app, support_credentials=True, origins=['address of your node app'])
parser = reqparse.RequestParser()
parser.add_argument('text')

# Load model
loaded_model = pickle.load(open('model.pkl', 'rb'))
loaded_model._make_predict_function()

# Load Tokenizer
with open('tokenize.pkl', 'rb') as handle:
    tokenize = pickle.load(handle)



@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict_api', methods=['POST'])
@cross_origin(supports_credentials=True)
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    index_list = tokenize.texts_to_matrix([data['data']])
    result = loaded_model.predict(np.array([index_list[0]]))[0]
    return jsonify(result.tolist())


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(debug=True)
