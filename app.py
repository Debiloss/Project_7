# 1. Library imports
import pickle

# from pydantic import BaseModel
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from sklearn.neighbors import NearestNeighbors

# 2. Create the app object
app = FastAPI(
    title="API-Scoring",
    description="Une API renvoyant une prédiction basé sur un modèle de scoring",
)

# Load dataset original
df_test = pd.read_csv(
    r"C:\Users\Sofia\OneDrive\Documents\OpenClassrooms\Project_7\home-credit-default-risk\application_test.csv")

# Load dataset modifié
df_test_modif = pd.read_csv(r"C:\Users\Sofia\OneDrive\Documents\OpenClassrooms\Project_7\Python\new_test.csv")

N_CUSTOMERS = 1000
N_NEIGHBORS = 20
MAIN_COLUMNS = ['SK_ID_CURR', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
                'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE',
                'AMT_INCOME_TOTAL',
                'DAYS_BIRTH', 'DAYS_EMPLOYED']

MAIN_COLUMNS_1 = ['SK_ID_CURR', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'NAME_FAMILY_STATUS',
                  'NAME_INCOME_TYPE', 'INCOME_CREDIT_PERC', 'ANNUITY_INCOME_PERC', 'DAYS_BIRTH', 'DAYS_EMPLOYED']

col = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR',
       'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'DAYS_EMPLOYED_PERC', 'INCOME_CREDIT_PERC',
       'INCOME_PER_PERSON', 'ANNUITY_INCOME_PERC', 'PAYMENT_RATE',
       'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
       'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
       'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
       'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'FLAG_MOBIL', 'FLAG_EMP_PHONE',
       'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
       'OCCUPATION_TYPE', 'REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START',
       'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION',
       'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
       'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
       'LIVE_CITY_NOT_WORK_CITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DEF_30_CNT_SOCIAL_CIRCLE',
       'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
       'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
       'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
       'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
       'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',
       'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
       'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

test_columns = df_test_modif.columns

# Load model with pickle
model = pickle.load(
    open(r'C:\Users\Sofia\OneDrive\Documents\OpenClassrooms\Project_7\modelisation\best_model.pickle', 'rb'))

# Deserialize SHAP explainer
explainer = pickle.load(open(r'C:\Users\Sofia\OneDrive\Documents\OpenClassrooms\Project_7\modelisation'
                             r'\lgbm_shap_explainer.pickle', 'rb'))

# Get datadrift html report
drift_report = open(r'C:\Users\Sofia\OneDrive\Documents\OpenClassrooms\Project_7\modelisation\data_drift_report.html',
                    'rb')


def prep_data(data, n_customers):
    """Mise en forme de la data à exploiter"""
    df = data.iloc[0:n_customers]

    return df


def prep_data_modif(data, n_neigbhors, n_customers):
    """Mise en forme de la data à exploiter"""
    df = data.iloc[0:n_customers]

    # Find nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=n_neigbhors, algorithm='ball_tree').fit(df)
    _, neighbors_indices = neighbors.kneighbors(df)

    # Compute shap values
    df1 = df.drop(axis=1, columns=['SK_ID_CURR'])
    shap_values = explainer(df1)

    # Create new df
    df = pd.DataFrame(df, columns=data.columns)

    return df, neighbors_indices, shap_values


# Prepare the reduced data and shap values
new_test = prep_data(df_test, N_CUSTOMERS)
new_test_modif, neighbors_indices, shap_values = prep_data_modif(df_test_modif, N_NEIGHBORS, N_CUSTOMERS)


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def home():
    """ API main page """
    return {"message": "Bienvenue sur l'API scoring"}


# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyIDsHere
# Retourne la liste des IDs présent pour la prédiction
@app.get('/ids')
def ids():
    """ Return the customers ids """
    return {'ids': new_test.head(N_CUSTOMERS).index.to_list()}


# Retourne un tableau avec les colonnes principales pour un client donné
@app.get("/columns/id={cust_id}")
def columns(cust_id: int):
    """ Return the customer main columns values """
    if cust_id not in new_test['SK_ID_CURR']:
        raise HTTPException(status_code=404, detail="Customer id not found")
    cust_main_df = new_test_modif.iloc[cust_id][MAIN_COLUMNS_1]
    return cust_main_df.to_json()


@app.get("/columns/mean")
def colmuns_mean():
    """ Return the main columns mean values """
    mean_df = new_test_modif[MAIN_COLUMNS_1].mean()
    return mean_df.to_json()


@app.get("/columns/neighbors/id={cust_id}")
def colmuns_neighbors(cust_id: int):
    """ Return the 20 nearest neighbors main columns mean values """
    if cust_id not in range(0, N_CUSTOMERS):
        raise HTTPException(status_code=404, detail="Customer id not found")
    neighbors_df = new_test_modif[MAIN_COLUMNS_1].iloc[neighbors_indices[cust_id]].mean()
    return neighbors_df.to_json()


# Prend en entrée un id client et renvoie une prédiction sur sa capacité à rembourser un pret
@app.get("/predict/id={cust_id}")
def predict(cust_id: int):
    """ Return the customer predictions of repay failure (class 1) """
    if cust_id not in range(0, N_CUSTOMERS):
        raise HTTPException(status_code=404, detail="Customer id not found")
    row = pd.DataFrame(new_test_modif[col].iloc[cust_id]).T
    proba = model.predict_proba(row)[0][1]  # prediction of class 1
    return {'proba': proba.tolist()}


@app.get("/shap")
def explain_all():
    """ Return all shap values """
    return {'values': shap_values.values.tolist(),
            'base_values': shap_values.base_values.tolist(),
            'features': explainer.feature_names}


@app.get("/shap/id={cust_id}")
def explain(cust_id: int):
    """ Return the customer shap values """
    if cust_id not in range(0, N_CUSTOMERS):
        raise HTTPException(status_code=404, detail="Customer id not found")
    return {'values': shap_values[cust_id].values.tolist(),
            'base_values': float(shap_values[cust_id].base_values),
            'features': explainer.feature_names}


@app.get("/importances")
def importances():
    """ Return the 15 top feature importances """
    imp_df = pd.DataFrame(data=model.feature_importances_, index=col, columns=['importances'])
    imp_df = imp_df.sort_values(by='importances', ascending=False).head(15)
    return imp_df.to_json()


@app.get("/datadrift")
def datadrift():
    """ Return the datadrift html report """
    return {'html': drift_report.read()}


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# tracer la distribution conditionnelle (ceux qui remboursent ou pas) avec filtres sur variables
# trace une ligne verticale le client que je score afin de voir sa distribution (en fonction voisins)
# sur la distribution indiquer sa localisation sur la distribution
# prédictions, probabilité de rembourser, le seuil de probabilités
# interprétabilité local et global
#
