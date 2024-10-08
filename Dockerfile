
####### 👇 SIMPLE SOLUTION (x86 and M1) 👇 ########
FROM python:3.8.12-buster

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY Stock_Price_Prediction_package_folder Stock_Price_Prediction_package_folder
COPY setup.py setup.py
RUN pip install .

COPY data data

COPY Makefile Makefile

CMD uvicorn Stock_Price_Prediction_package_folder.api:app --host 0.0.0.0 --port $PORT
