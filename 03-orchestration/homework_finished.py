import os
import joblib
import pyarrow.parquet as pq
import pandas as pd
from datetime import datetime

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import task, flow, get_run_logger

@task
def get_paths(date):
    # define directory for dataset
    file_location = "../dataset/"
    
    # parse date to datetime type
    if date:
        processed_date = datetime.strptime(date, "%Y-%m-%d")
    else:
        processed_date = datetime.today()
    
    # train and validation data month
    year = processed_date.year
    train_date = processed_date.month - 2
    val_date = processed_date.month - 1
    
    # check if the files exist or not
    train_path = f"{file_location}fhv_tripdata_{year}-0{train_date}.parquet"
    val_path = f"{file_location}fhv_tripdata_{year}-0{val_date}.parquet"
    
    return train_path, val_path

@task
def read_dataframe(path: str):
    return pd.read_parquet(path)

@task
def preprocess_features(df, categorical, train=True):
    # define the logger to print something
    logger = get_run_logger()
    
    # lowercase columns
    df.columns = df.columns.str.lower()
    
    # new feature: duration
    df['duration'] = df['dropoff_datetime'] - df['pickup_datetime']
    df['duration'] = df['duration'].dt.total_seconds() / 60
    
    # print the mean of duration in minutes
    mean_duration = df['duration'].mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration:.3f}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration:.3f}")
    
    # filtering the duration
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

@task
def train_model(df, categorical):
    # define the logger to print something
    logger = get_run_logger()
    
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    
    X_train = dv.fit_transform(train_dicts) 
    y_train = df['duration'].values
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_train)
    rmse = mean_squared_error(y_train, y_pred, squared=False)
    
    # print training shape, dummy variables count, train rmse
    logger.info(f"X_train size: {X_train.shape}")
    logger.info(f"Dict Vectorizer has {len(dv.feature_names_)} features")
    logger.info(f"RMSE of training data: {rmse:.3f}")

    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    # define the logger to print something
    logger = get_run_logger()
    
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    
    y_pred = lr.predict(X_val)
    y_val = df['duration'].values

    rmse = mean_squared_error(y_val, y_pred, squared=False)
    
    # print validation shape, dummy variables count, validation rmse
    logger.info(f"X_val size: {X_val.shape}")
    logger.info(f"Dict Vectorizer has {len(dv.feature_names_)} features")
    logger.info(f"RMSE of validation data: {rmse:.3f}")
    
    return

@flow
def main(date):
    train_path, val_path = get_paths(date).result()
    
    df_train = read_dataframe(train_path)
    df_val = read_dataframe(val_path)#, wait_for=[train_path, val_path])
    
    categorical = ['pulocationid', 'dolocationid']
    df_train_processed = preprocess_features(df_train, categorical, train=True)
    df_val_processed = preprocess_features(df_val, categorical, train=False)
    
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)
    
    if date is None:
        date = datetime.today.strftime("%Y-%m-%d")
    
    with open(f'./models/model-{date}.bin', 'wb') as f_out:
        joblib.dump(dv, f_out)    
    
    with open(f'./models/dv-{date}.b', 'wb') as f_out:
        joblib.dump(dv, f_out)

#main("2021-03-01")

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    flow=main,
    name="model_training_hw",
    schedule=CronSchedule(cron="0 9 15 * *"),
    flow_runner=SubprocessFlowRunner(),
)