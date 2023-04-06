# Chargement du modèle avec MlFlow ou Pickle
import pickle

# Bibliothèque essentiel python
import pandas as pd
# Serveur ASGI (Asynchronous Server Gateway Interface)
import uvicorn
# Framework FastAPI pour la création de l'API sous python
from fastapi import FastAPI, HTTPException
# Validation des données entrantes
from pydantic import BaseModel

# Load the MLflow model
# loaded_model = mlflow.pyfunc.load_model("path/to/your/mlflow/model")

# Load model with pickle
model = pickle.load(
    open(r'C:\Users\Sofia\OneDrive\Documents\OpenClassrooms\Project_7\modelisation\best_model.pickle', 'rb'))

# Deserialize SHAP explainer
# explainer = pickle.load(open('./models/_shap_explainer.pckl', 'rb'))

N_CUSTOMERS = 1000
N_NEIGHBORS = 20
MAIN_COLUMNS = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
                'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE',
                'AMT_INCOME_TOTAL', 'PAYMENT_RATE',
                'DAYS_BIRTH', 'DAYS_EMPLOYED']
CUSTOM_THRESHOLD = 0.7

# Load dataset original
df_test = pd.read_csv(
    r"C:\Users\Sofia\OneDrive\Documents\OpenClassrooms\Project_7\home-credit-default-risk\application_test.csv")
# Load dataset modifié
df_test_modif = pd.read_csv(r"C:\Users\Sofia\OneDrive\Documents\OpenClassrooms\Project_7\new_test.csv")
test_columns = df_test.columns

# Compute shap values
# shap_values = explainer(df_test_modif)

# Create an instance of the FastAPI
app = FastAPI(
    title="API-Scoring",
    description="Une API renvoyant une prédiction basé sur un modèle de scoring",
)

# Obtenir une liste de toutes les variables et leurs types de données
variable_types = df_test_modif.dtypes


# Créer une classe de données d'entrée pour la prédiction
class PredictionInput(BaseModel):
    for variable, dtype in variable_types.items():
        if dtype == 'object':
            # Champ de validation pour les variables de type objet (texte)
            locals()[variable]: str
        elif dtype == 'int64':
            # Champ de validation pour les variables de type entier
            locals()[variable]: int
        elif dtype == 'float64':
            # Champ de validation pour les variables de type flottant (décimal)
            locals()[variable]: float


# Page Home: Affichage d'un message de bienvenue
@app.get("/")
def home():
    """ API main page """
    return {"message": "Bienvenue sur l'API scoring"}


# Retourne la liste des IDs présent pour la prédiction
@app.get("/ids")
def ids():
    """ Return the customers ids """
    return {'ids': df_test.head(N_CUSTOMERS).home.to_list()}


# Retourne un tableau avec les colonnes principales pour un client donné
@app.get("/columns/id={cust_id}")
def columns(cust_id: int):
    """ Return the customer main columns values """
    if cust_id not in df_test['SK_ID_CURR']:
        raise HTTPException(status_code=404, detail="Customer id not found")
    cust_main_df = df_test.iloc[cust_id][MAIN_COLUMNS]
    return cust_main_df.to_json()


# Prend en entrée un id client et renvoie une prédiction sur sa capacité à rembourser un pret
@app.get("/predict/id={cust_id}")
def predict(data: PredictionInput, cust_id: int):
    df = pd.DataFrame(data, columns=test_columns)
    prediction = model.predict_proba(df.iloc[cust_id])[0][1]  # predictions class 1
    return {"prediction": prediction.to_list()}


# Renvoie l'importance global des variables pour le dataset entier
# @app.get("/shap")
# def explain_all():
#    """ Return all shap values """
#    return {'values': shap_values.values.tolist(),
#            'base_values': shap_values.base_values.tolist(),
#            'features': explainer.feature_names}


# Renvoie l'importance global des variables pour un client donné
# @app.get("/shap/id={cust_id}")
# def explain(cust_id: int):
#    """ Return the customer shap values """
#    if cust_id not in range(0, N_CUSTOMERS):
#        raise HTTPException(status_code=404, detail="Customer id not found")
#    return {'values': shap_values[cust_id].values.tolist(),
#            'base_values': float(shap_values[cust_id].base_values),
#            'features': explainer.feature_names}


# Renvoie l'importance local des variables en fonction du model utilisé pour la prédiction
@app.get("/importances")
def importances():
    """ Return the 15 top feature importances """
    imp_df = pd.DataFrame(data=model.feature_importances_, index=test_columns, columns=['importances'])
    imp_df = imp_df.sort_values(by='importances', ascending=False).head(15)
    return imp_df.to_json()


# Renvoie le rapport datadrift
# @app.get("/datadrift")
# def datadrift():
#    """ Return the datadrift html report """
#    return {'html': drift_report.read()}


# Run the API with Uvicorn
if __name__ == "__main__":
    uvicorn.run("scoring_api:app", reload=True, host="127.0.0.1", port=8000)

# dashbord renvoie la probabilité de prédiction,
# streamlit pour afficher les importance local et global

# finir l'api web en parallèle du dashbord. Api web -> back, Dashbord -> Front
# streamlit et connaisance des variables du dataset
# datadrift, comment le faire ? etc..
# test unitaire et déploiement
# Revoir le travail sur les variables (preprocessing) ainsi que finition de la modélisation
# Git + note méthodologique + presentation
