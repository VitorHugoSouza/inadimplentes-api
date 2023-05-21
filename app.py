import numpy as np
from flask import Flask, request, render_template, jsonify
from joblib import load
import json
from flask_cors import CORS

app = Flask(__name__, template_folder='template') 
CORS(app)
model = load('melhor_modelo.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data = request.get_json(force=True)    
    valores = [int(data['Idade']),int(data['Sexo']),int(data['EstadoCivil']),int(data['RendaMensal'])]    
    int_features = [int(x) for x in valores]    
    final_features = [np.array(int_features)]    
    prediction = model.predict(final_features)    
    prediction_prob = model.predict_proba(final_features)    
    output = round(prediction[0], 2)    
   
    probabilidade_pagar     = np.array(prediction_prob[:,0])
    probabilidade_nao_pagar = np.array(prediction_prob[:,1]) 
    
    resposta = { 'resultado': int(output), 
                'probabilidade_pagar': float(probabilidade_pagar), 
                'probabilidade_nao_pagar': float(probabilidade_nao_pagar)
                }
    
    return jsonify(resposta)
    #return render_template('index.html', prediction_text='(0 - Não inadimplente ou 1 - Inadimplente) : {} {}'.format(output, prediction_prob))
    #return json.dumps({'inadimplente':output, 'probabilidade':str(prediction_prob)}, default=int)

    #return '(0 - Não inadimplente ou 1 - Inadimplente) : {} {}'.format(output, prediction_prob)

if __name__ == "__main__":
    app.run(debug=True)
