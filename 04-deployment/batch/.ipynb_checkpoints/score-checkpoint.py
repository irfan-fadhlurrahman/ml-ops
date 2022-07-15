import pandas as pd
import mlflow
import joblib
import uuid
import sys

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from prefect import task, flow, get_run_logger
from prefect.context import get_run_context
from datetime import datetime
from dateutil.relativedelta import relativedelta

def load_dataset(path: str):
    df = pd.read_parquet(path)
    return df

def generate_uuids(length):
    return [str(uuid.uuid4()) for i in range(length)]

def prepare_features(df: pd.DataFrame):
    # duration
    df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    df['duration'] = df['duration'].dt.total_seconds() / 60
    
    # pickup and dropoff location
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    df.loc[:, "PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
    
    # every filter must be at the end of the function
    return df[(df['duration'] >= 1) & (df['duration'] <= 60)]

def load_model(run_id):
    # select the model to use by choosing run_id
    artifact_root_dir = '../web-service-mlflow'
    logged_model = f'{artifact_root_dir}/mlruns/1/{run_id}/artifacts/model'

    # load model as a PyFuncModel and predict the dicts
    return mlflow.pyfunc.load_model(logged_model)

def save_results(df, y_pred, run_id, output_file):
    # create the result dataframe for scoring job
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id
    
    # save the result as parquet file
    df_result.to_parquet(output_file, index=False)


@task
def apply_model(input_file, run_id, output_file):
    logger = get_run_logger()
    
    # load the dataset
    logger.info(f"reading the dataset from {input_file}....")
    df = load_dataset(input_file)
    
    # generate artificial ride_id
    df['ride_id'] = generate_uuids(length=len(df))
    
    # preprocess the dataframe
    features = ['PU_DO', 'trip_distance']
    df = prepare_features(df)
    dicts= df[features].to_dict(orient='records')
    
    # predict
    logger.info(f"load the model with RUN_ID={run_id}....")
    model = load_model(run_id)
    
    logger.info(f"apply the model....")
    y_pred = model.predict(dicts)
    
    logger.info(f"saved the output file to {output_file}....")
    save_results(df, y_pred, run_id, output_file)
    return output_file

def get_paths(run_date, taxi_type, run_id):
    prev_month = run_date - relativedelta(months=1)
    year = prev_month.year
    month = prev_month.month 

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'./output/{taxi_type}/{year:04d}-{month:02d}.parquet'

    return input_file, output_file


@flow
def ride_duration_prediction(
        taxi_type: str,
        run_id: str,
        run_date: datetime = None):
    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow_run.expected_start_time
    
    input_file, output_file = get_paths(run_date, taxi_type, run_id)

    apply_model(
        input_file=input_file,
        run_id=run_id,
        output_file=output_file
    )
    
def run():
    # dataset information
    taxi_type = sys.argv[1] #'green'
    year = int(sys.argv[2]) # 2021
    month = int(sys.argv[3]) #3
    
    # model artifact
    run_id = sys.argv[4] # 'd3b600b0d89e4e95b27b5e333e4d3c01'
    
    # result
    ride_duration_prediction(
        taxi_type=taxi_type,
        run_id=run_id,
        run_date=datetime(year=year, month=month, day=1)
    )

if __name__ == '__main__':
    run()