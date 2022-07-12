import joblib 
from flask import Flask, request, jsonify

model_path = "./2022-07-12_lin_reg.bin"
with open(model_path, 'rb') as file_in:
    dv, model = joblib.load(file_in)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features
    
def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return preds[0]


app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)
    
    result = {
        "duration": float(pred)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)