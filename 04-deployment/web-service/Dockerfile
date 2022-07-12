# notes: run the following command on terminal
# build the image
#    docker build -t ride-duration-prediction-service:v1 .
# run the container and delete the image
#    docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1

# use slim version to have smaller size of python
FROM python:3.9.7-slim

# update python pip to avoid error while installing libraries
# i.e. install xgboost
RUN pip install -U pip
RUN pip install pipenv

# create new folder
WORKDIR /app

# copy dependencies
COPY [ "Pipfile", "Pipfile.lock", "./" ]

# system and deploy args mean no to create new environment
# because Docker already isolated this configuration from our OS
RUN pipenv install --system --deploy
COPY [ "predict.py", "2022-07-12_lin_reg.bin", "./" ]

# configure the deployed server
EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]