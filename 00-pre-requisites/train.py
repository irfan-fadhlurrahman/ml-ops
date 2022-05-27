import pandas as pd
import numpy as np
import inflection
import dill

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

SEED = 0

def load_dataset(path):
    """
    Return the dataset in pandas dataframe format with lowercase columns
    and each column name seperated by underscore.
    
    Args:
    * path, str: file directory or file link of dataset
    
    Output:
    * pd.DataFrame
    
    Notes:
    * This function is used to read a parquet format.
      If the format is not parquet, change the first line of code.
      i.e. pd.read_csv.
    
    """
    df = pd.read_csv(path)
    df.columns = [inflection.underscore(var) for var in list(df.columns)]    
    return df

def parse_to_numeric(df, var='total_charges'):
    """
    Parse the numerical variables that are still in object dtype. 
    
    Args:
    * df, pd.DataFrame: the dataset
    * var (default: 'total_charges'), str: a variable to be parsed
    * fillna (default: True), bool: option to fill missing values or not
    Return:
    * pd.DataFrame
    
    """
    df = df.copy()
    df[var] = pd.to_numeric(df[var], errors='coerce')
    return df

def split_dataset(df, target='churn', test_size=0.2, seed=SEED):
    """
    Return the train and validation/test set with specified split ratio.
    
    Args:
    * df, pd.DataFrame: the dataset to be splitted
    * target (default: 'churn'), str: the target variable
    * test_size (default: 0.2), float: the ratio of test size after splitting
    * seed (default: SEED), int: random number for reproducibility
    
    Output:
    * train, val (optional: test), pd.DataFrame: training and validation/test sets
    
    Notes:
    * The purpose of splitted is to avoid data leakage and
      to make an hold-out dataset for testing the model performance.
    * Use strafied sampling to make training and validation/test data 
      have similar distribution.

    """
    return train_test_split(
        df.fillna(0),
        test_size=test_size,
        random_state=seed,
        stratify=df[target]
    )

def train_the_model(train, val, target='churn', seed=SEED):
    """
    Return the trained model and print the chosen evaluatoin metric.
    
    Args:
    * train, pd.DataFrame: training set
    * val, pd.DataFrame: either validation or test set
    * target (default: 'churn'), str: the target variable
    * seed (default: SEED), int: random number for reproducibility
    
    Output:
    * model, .bin: the trained model
    
    Notes:
    * Do not forget to make features and target matrix, X and y
    * If using DictVectorizer, convert features matrix into dictionary.
    * Use pipeline instead of manual data preprocessing and modelling.
    * Use Logistic Regression to train the model for building a simple
      model.
      
    """
    X_train = train.drop(target, axis=1)
    X_val = val.drop(target, axis=1)
    
    # use FunctionTransformer to automatically convert features matrix to dictionary
    #def matrix_to_dict(X): return X.to_dict(orient='records')
    
    y_train = train[target]
    y_val = val[target]
    
    model = make_pipeline(
        FunctionTransformer(lambda X: X.to_dict(orient='records')),
        DictVectorizer(sparse=False),
        MinMaxScaler(),
        LogisticRegression(random_state=seed, 
                           solver='liblinear', 
                           class_weight='balanced')
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    print("Classification Report for validation set\n\n", classification_report(y_val, y_pred))
    print(f"ROC AUC for the validation set: {roc_auc_score(y_val, y_proba):.3f}\n")
    
    return model

def save_the_model(model, file_name="logreg"):
    """
    Save the machine learning model that contains preprocessor and regressor/classifier.
    
    Args:
    * model, .bin: model pipeline to predict unseen data later
    * file_name (default: "logreg"): name of the used classifier
    
    Output:
    * None
    
    """
    with open(f"./model/{file_name}.bin", 'wb') as file_out:
         dill.dump(model, file_out)
    
def load_the_model(path):
    """
    Load the trained model from directory
    
    Args:
    * path, str: trained model directory
    
    Output:
    * model, ... : a loaded model
    
    """
    with open(f"{path}", 'rb') as file_in:
         model = dill.load(file_in)
    
    return model

def main():
    # data preparation
    df = load_dataset("./dataset/telco-customer-churn.csv")
    df = parse_to_numeric(df, var='total_charges')
    df['churn'] = df['churn'].map({"No": 0, "Yes": 1})
    
    # set up a validation framework
    full_train, test = split_dataset(df, target='churn', test_size=0.2)
    train, val = split_dataset(full_train, target='churn', test_size=0.25)
    
    # train the model
    model = train_the_model(train, val, target='churn')
    
    # save the model
    save_the_model(model, file_name="logreg-v2")
    
    # load the model and use it to predict test set
    trained_model = load_the_model("./model/logreg.bin")
    
    X_test = test.drop("churn", axis=1)
    y_test = test["churn"]
    
    y_pred = trained_model.predict(X_test)
    y_proba = trained_model.predict_proba(X_test)[:, 1]
    print("Classification Report for test set\n\n", classification_report(y_test, y_pred))
    print(f"ROC AUC for the test set: {roc_auc_score(y_test, y_proba):.3f}\n")
    
main()