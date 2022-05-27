import dill
import pandas as pd

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'logreg-v2.bin'
with open(f"./{model_file}", 'rb') as file_in:
     model = dill.load(file_in)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    
    X = pd.DataFrame([customer])
    y_proba = model.predict_proba(X)[0, 1]
    churn = y_proba >= 0.5
    
    result = {
        "churn_probability": float(y_proba),
        "churn": bool(churn)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)