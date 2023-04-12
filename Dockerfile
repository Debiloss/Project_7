FROM python:3.10.11-slim as build
EXPOSE 8000
WORKDIR /main
COPY / .
COPY /best_model.pickle ./best_model.pickle
COPY /lgbm_shap_explainer.pickle ./lgbm_shap_explainer.pickle
COPY /data_drift_report.html ./data_drift_report.html
COPY /data_model_test.csv ./data_model_test.csv
COPY /test_info.csv ./test_info.csv
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y libgomp1
CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]