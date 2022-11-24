import numpy as np
from flask import Flask, request, render_template
from joblib import load

app = Flask(__name__,template_folder='template') #Initialize the flask App
model = load('melhor_modelo.joblib')

# @app.route('/')
# def home():
#    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    prediction_prob = model.predict_proba(final_features)
    output = round(prediction[0], 2)

    return '(0 - Não inadimplente ou 1 - Inadimplente) : {} {}'.format(output, prediction_prob)

if __name__ == "__main__":
    app.run(debug=True)