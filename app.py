from flask import Flask, request, jsonify
import mlflow
import pandas as pd
from functions.experiments import Experiments

app = Flask(__name__)

@app.route('/', methods=['GET'])
def main(): return 'API FLASK OK', 200

@app.route('/api/model/<id>', methods=['POST'])
def load_model_by_id(id):
    data = request.get_json()
    logged_model = f'runs:/{id}/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    response = loaded_model.predict(pd.DataFrame({
        "sepal length (cm)": data.get('sepal length (cm)'),
        "sepal width (cm)": data.get('sepal width (cm)'),
        "petal length (cm)": data.get('petal length (cm)'),
        "petal width (cm)": data.get('petal width (cm)')
    }, index=[0]))
        
    return jsonify({ 'class': int(response[0]) })

@app.route('/api/experiments', methods=['GET'])
def get_all_experiments():
    experiments = Experiments()
    experiments = experiments.get_all_experiments()
    return experiments

if __name__ == '__main__':
    app.run(debug=True, port=8080)
    
    