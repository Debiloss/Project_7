FROM python:3.10.11-slim as build
EXPOSE 8000
WORKDIR /API/main
COPY ./API /API
COPY /API/best_model.pickle ./API/best_model.pickle
COPY /API/lgbm_shap_explainer.pickle ./API/lgbm_shap_explainer.pickle
COPY /API/data_drift_report.html ./API/data_drift_report.html
COPY /API/data_model_test.csv ./API/data_model_test.csv
COPY /API/test_info.csv ./API/test_info.csv
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]