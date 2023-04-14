from fastapi.testclient import TestClient
from fastapi import status
from main import app
import numpy as np

client = TestClient(app=app)


def test_home():
    response = client.get('/')
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"message": "Bienvenue sur l'API scoring"}


def test_ids():
    response = client.get('/ids')
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"ids": list(np.arange(0, 1000, 1))}


def test_predict():
    cust_id = 2
    response = client.get(f'/predict/id={cust_id}')
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"proba": 0.1282767443124775}

