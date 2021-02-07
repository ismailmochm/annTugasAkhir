from flask import Flask, render_template, request
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras import backend as K

app = Flask(__name__, template_folder='templates')


@app.route('/')
def student():
    return render_template("index.html")


def ValuePredictor(to_predict_list):
    K.clear_session()
    model = load_model('model.h5')
    to_predict = np.array(to_predict_list).reshape(1, 6)
    result = model.predict(to_predict)
    K.clear_session()
    return result[0]


@app.route('/predict', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = float(ValuePredictor(to_predict_list))
        result = ValuePredictor(to_predict_list)
        if result > 0.5:
            result ='cerai'
        else:
            result = 'tidak cerai'
        return render_template("index.html", result=result)


if __name__ == '__main__':
    app.run(debug=True)
