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