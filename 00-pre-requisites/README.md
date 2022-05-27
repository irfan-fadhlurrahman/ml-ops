## Machine Learning Operations (MLOps): Pre-requisites
### Deployed the machine learning model by using Flask and Docker

`Python version: 3.7`
`virtual environment: Pipfile`

The goal of this project to train and save the model then use the model in a web service. For this project, we will use Telco Customer Churn dataset from Kaggle. 
To save model I will use `dill` instead `pickle` because I tried using `pickle` but something wrong occured in my saved model due to `pickle` cannot save lambda function in the model pipeline.
