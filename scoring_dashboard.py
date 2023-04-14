import streamlit as st
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import altair as alt
import shap
import requests
import names
import json
import seaborn as sns

# API_URL = 'http://127.0.0.1:8000/'
API_URL = 'https://apicreditscoring.herokuapp.com/'

# Timeout for requests (connect, read)
TIMEOUT = (5, 30)

# Prediction classes
CLASSES_NAMES = ['REPAY SUCCESS', 'REPAY FAILURE']
CLASSES_COLORS = ['green', 'red']


# Functions

# Créer des noms aléatoires pour la liste de clients
@st.cache_data
def create_customer_names(cust_numbers):
    """ Create array of random names """
    return [names.get_full_name() for _ in range(cust_numbers)]


# Retourne la liste des IDs de nos clients présents
@st.cache_data
def get_cust_ids():
    """ Get list of customers ids """
    response = requests.get(API_URL + "ids/", timeout=TIMEOUT)
    content = json.loads(response.content)
    return content['ids']


@st.cache_data
def get_gender():
    """ Get gender of customers """
    response = requests.get(API_URL + "gender/", timeout=TIMEOUT)
    content = json.loads(json.loads(response.content))
    return pd.Series(content)


@st.cache_data
def get_education():
    """ Get education type of customers """
    response = requests.get(API_URL + "education/", timeout=TIMEOUT)
    content = json.loads(json.loads(response.content))
    return pd.Series(content)


@st.cache_data
def get_age():
    """ Get age of customers """
    response = requests.get(API_URL + "age/", timeout=TIMEOUT)
    content = json.loads(json.loads(response.content))
    return pd.Series(content)


@st.cache_data
def get_income():
    """ Get income of customers """
    response = requests.get(API_URL + "income/", timeout=TIMEOUT)
    content = json.loads(json.loads(response.content))
    return pd.Series(content)


@st.cache_data
def get_payment():
    """ Get income of customers """
    response = requests.get(API_URL + "payment/", timeout=TIMEOUT)
    content = json.loads(json.loads(response.content))
    return pd.Series(content)


@st.cache_data
def get_credit_perc():
    """ Get income credit perc of customers """
    response = requests.get(API_URL + "credit_perc/", timeout=TIMEOUT)
    content = json.loads(json.loads(response.content))
    return pd.Series(content)


@st.cache_data
def get_income_perc():
    """ Get income perc of customers """
    response = requests.get(API_URL + "income_perc/", timeout=TIMEOUT)
    content = json.loads(json.loads(response.content))
    return pd.Series(content)


@st.cache_data
def get_target():
    """ Get target of customers """
    response = requests.get(API_URL + "target/", timeout=TIMEOUT)
    content = json.loads(json.loads(response.content))
    return pd.Series(content)


# Retourne les colonnes principales pour un client donné
@st.cache_data
def get_cust_columns(cust_id):
    """ Get customer main columns """
    response = requests.get(API_URL + "columns/id=" + str(cust_id), timeout=TIMEOUT)
    content = json.loads(json.loads(response.content))
    return pd.Series(content)


@st.cache_data
def get_columns_mean():
    """ Get customers main columns mean values """
    response = requests.get(API_URL + "columns/mean", timeout=TIMEOUT)
    content = json.loads(json.loads(response.content))
    return pd.Series(content)


@st.cache_data
def get_columns_neighbors(cust_id):
    """ Get customers neighbors main columns mean values """
    response = requests.get(API_URL + "columns/neighbors/id=" + str(cust_id), timeout=TIMEOUT)
    content = json.loads(json.loads(response.content))
    return pd.Series(content)


# Retourne la valeur de la prédiction
@st.cache_data
def get_predictions(cust_id):
    """ Get customer prediction (class 1 : repay failure) """
    response = requests.get(API_URL + "predict/id=" + str(cust_id), timeout=TIMEOUT)
    content = json.loads(response.content)
    return content


# Retourne les importances globales des variables du dataset
@st.cache_data
def get_shap_values():
    """ Get all customers SHAP values """
    response = requests.get(API_URL + "shap", timeout=TIMEOUT)
    content = json.loads(response.content)
    explanation = shap.Explanation(np.asarray(content['values']),
                                   np.asarray(content['base_values']),
                                   feature_names=content['features'])
    return explanation


# Retourne les importances globales des variables pour un client donné
@st.cache_data
def get_shap_explanation(cust_id):
    """ Get customer SHAP explanation """
    response = requests.get(API_URL + "shap/id=" + str(cust_id), timeout=TIMEOUT)
    content = json.loads(response.content)
    explanation = shap.Explanation(np.asarray(content['values']),
                                   content['base_values'],
                                   feature_names=content['features'])
    return explanation


# Retourne les importances locales des variables du dataset
@st.cache_data
def get_feature_importances():
    """ Get feature importance """
    response = requests.get(API_URL + "importances", timeout=TIMEOUT)
    content = json.loads(json.loads(response.content))
    return pd.DataFrame(content)


# Retourne les importances locales des variables en fonction du modele utilisé pour la prédiction
def model_importances_chart():
    """ Return altair chart of feature importances """
    imp_df = get_feature_importances()
    imp_sorted = imp_df.sort_values(by='importances', ascending=False)
    imp_chart = alt.Chart(imp_sorted.reset_index(), title="Top 15 feature importances").mark_bar().encode(
        x='importances',
        y=alt.Y('index', sort=None, title='features'))
    return imp_chart


# 
def st_shap(plot, height=None):
    """ Create a shap html component """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


# Retourne le rapport data drift
@st.cache_data
def get_datadrift_report():
    """ Get data drift html report """
    response = requests.get(API_URL + "datadrift", timeout=TIMEOUT)
    content = json.loads(response.content)
    return content['html']


# ====================================================================
# HEADER - TITRE
# ====================================================================
html_header = """
    <head>
        <title>Application Dashboard Crédit Score</title>
        <meta charset="utf-8">
        <meta name="keywords" content="Home Crédit Group, Dashboard, prêt, crédit score">
        <meta name="description" content="Application de Crédit Score - dashboard">
        <meta name="author" content="Sofiane Mimeche">
        <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>             
    <h1 style="font-size:300%; color:Crimson; font-family:Arial"> Prêt à dépenser <br>
        <h2 style="color:Gray; font-family:Georgia"> DASHBOARD</h2>
        <hr style= "  display: block;
          margin-top: 0;
          margin-bottom: 0;
          margin-left: auto;
          margin-right: auto;
          border-style: inset;
          border-width: 1.5px;"/>
     </h1>
"""
st.set_page_config(page_title="Prêt à dépenser - Dashboard", page_icon="", layout="wide")
st.markdown('<style>body{background-color: #fbfff0}</style>', unsafe_allow_html=True)
st.markdown(html_header, unsafe_allow_html=True)

# Cacher le bouton en haut à droite
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

# Suppression des marges par défaut
padding = 1
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

# Settings on sidebar
st.sidebar.subheader("Settings")
# Select the prediction threshold
pred_thresh = st.sidebar.slider("Prediction threshold : ", 0.15, 0.55, value=0.50, step=0.01,
                                help="Threshold of the prediction for class 1 : repay failure (standard=0.5)")
# Select type of explanation
shap_plot_type = st.sidebar.radio("Select the plot type :", ('Waterfall', 'Bar'),
                                  help="Type of plot for the SHAP explanation")
# Select source of feature importance
feat_imp_source = st.sidebar.radio("Feature importances source :", ('LGBM', 'SHAP'),
                                   help="Feature importances computed from the LGBM model or from the SHAP values")

# Create tabs
tab_single, tab_all, tab_inf = st.tabs(["Single customer", "All customers", "Information customers"])

# General tab
with tab_inf:
    expander = st.expander("About the customers..")
    expander.write("Some informations about the Gender and type Education")

    st.subheader("Distribution Gender")
    df = get_gender().value_counts()
    df = df.rename('Gender')
    st.bar_chart(data=df, use_container_width=True)
    st.write("")

    st.subheader("Distribution Education")
    df1 = get_education().value_counts()
    df1 = df1.rename('Education')
    st.bar_chart(data=df1, use_container_width=True)
    st.write("")

    expander = st.expander("About the customers..")
    expander.write("Some informations about the Income total and Age per Gender")

    st.subheader("Distribution Income/Age per Gender")
    df_gender = get_gender()
    df_age = get_age()
    df_income = get_income()
    source = pd.DataFrame(columns=['Gender', 'Income', 'Age'])
    source['Gender'] = df_gender
    source['Income'] = df_income
    source['Age'] = df_age
    chart = alt.Chart(source).mark_circle().encode(
        x="Income",
        y="Age",
        color="Gender",
    ).interactive()
    st.altair_chart(chart, theme="streamlit", use_container_width=True)
    st.write("")

    expander = st.expander("About the customers..")
    expander.write("Payment is the variable payment rate. AMT_ANNUITY/AMT_CREDIT")

    st.subheader("Distribution Income/Age per Gender with Education & Payment informations")
    df_gender = get_gender()
    df_age = get_age()
    df_income = get_income()
    df_education = get_education()
    df_payment = get_payment()
    source = pd.DataFrame(columns=['Gender', 'Income', 'Age', 'Education', 'Payment'])
    source['Gender'] = df_gender
    source['Income'] = df_income
    source['Age'] = df_age
    source['Education'] = df_education
    source['Payment'] = df_payment*5

    scale = alt.Scale(
        domain=["M", "F"],
        range=["#9467bd", "#1f77b4"],
    )
    color = alt.Color("Gender:N", scale=scale)
    brush = alt.selection_interval(encodings=["x"])
    click = alt.selection_multi(encodings=["color"])

    points = (
        alt.Chart()
        .mark_point()
        .encode(
            alt.X("Income:N", title="Income total"),
            alt.Y(
                "Age:Q",
                title="Age",
                scale=alt.Scale(domain=[20, 70]),
            ),
            color=alt.condition(brush, color, alt.value("lightgray")),
            size=alt.Size("Payment:Q", scale=alt.Scale(range=[15, 65])),
        )
        .properties(width=550, height=300)
        .add_selection(brush)
        .transform_filter(click)
    )

    bars = (
        alt.Chart()
        .mark_bar()
        .encode(
            x="count()",
            y="Education:N",
            color=alt.condition(click, color, alt.value("lightgray")),
        )
        .transform_filter(brush)
        .properties(
            width=550,
        )
        .add_selection(click)
    )

    chart = alt.vconcat(points, bars, data=source)

    st.altair_chart(chart, theme="streamlit", use_container_width=True)

    expander = st.expander("About the customers..")
    expander.write("Some informations about % variables")

    st.subheader("Informations % variables")
    df_perc = pd.DataFrame(columns=["Income/Credit", "Annuity/Income", "Payment"])
    df_perc["Income/Credit"] = get_credit_perc()
    df_perc["Annuity/Income"] = get_income_perc()
    df_perc["Payment"] = get_payment()
    cust_names = create_customer_names(df_perc.shape[0])
    df_perc.index = cust_names
    st.line_chart(df_perc)
    st.write("")


with tab_all:
    st.subheader("Feature importances (" + feat_imp_source + ")")
    st.write("")

    if feat_imp_source == 'LGBM':
        # Display LGBM feature importance
        st.altair_chart(model_importances_chart(), use_container_width=True)
        expander = st.expander("About the feature importances..")
        expander.write("The feature importances displayed is computed from the trained LGBM model.")

    else:
        # Display SHAP feature importance
        shap_values = get_shap_values()
        fig, _ = plt.subplots()
        fig.suptitle('Top 15 feature importances (test set)', fontsize=18)
        shap.summary_plot(shap_values, max_display=15, plot_type='bar', plot_size=[12, 6], show=False)
        st.pyplot(fig)
        expander = st.expander("About the feature importances..")
        expander.write(
            "The feature importances displayed is computed from the SHAP values of the new customers. (test data)")

    # Display the datadrift report 
    st.subheader("Data drift report")
    components.html(get_datadrift_report(), height=1000, scrolling=True)
    expander = st.expander("About the data drift...")
    expander.write("The data drift report shows the drift between the data used to train the model \
                    and the customers data used in this application (test data).")

# Specific customer tab
with tab_single:
    # Get customer ids
    cust_ids = get_cust_ids()
    cust_names = create_customer_names(len(cust_ids))

    # Select the customer
    cust_select_id = st.selectbox(
        "Select the customer",
        cust_ids,
        format_func=lambda x: str(x) + " - " + cust_names[x])

    # Display the columns
    st.subheader("Customer information")
    cust_df = get_cust_columns(cust_select_id).rename(cust_names[cust_select_id])
    neighbors_df = get_columns_neighbors(cust_select_id).rename("Neighbors average")
    mean_df = get_columns_mean().rename("Customers average")
    st.dataframe(pd.concat([cust_df, neighbors_df, mean_df], axis=1))

    # Display prediction
    st.subheader("Customer prediction")
    predictions = get_predictions(cust_select_id)

    pred = (predictions['proba'] >= pred_thresh)
    pred_text = "**:" + CLASSES_COLORS[pred] + "[" + CLASSES_NAMES[pred] + "]**"
    st.markdown("The model prediction is " + pred_text)
    probability = 1 - round(predictions['proba'], 2)  # probability of repay (class 0)
    delta = probability - pred_thresh
    st.metric(label="Probability to repay", value=probability, delta=round(delta, 2))

    # Display some information
    expander = st.expander("About the classification model...")
    expander.write("The prediction was made using a LGBM classification model.")
    expander.write("The model threshold can be modified in the settings. \
                    The default threshold predict a repay failure when probability is lower or equal to 0.5. \
                    The best optimized threshold predict a repay failure when probability is lower or equal to 0.5")

    target = 1 - get_target()
    fig, ax = plt.subplots()
    sns.histplot(target, ax=ax, kde=True)
    plt.axvline(x=pred_thresh, color='r')
    plt.axvline(x=predictions, color='purple', linestyle='--')
    st.pyplot(fig)

    # Display shap force plot
    shap_explanation = get_shap_explanation(cust_select_id)
    st_shap(shap.force_plot(shap_explanation))

    # Display shap bar/waterfall plot
    fig, _ = plt.subplots()
    if shap_plot_type == 'Waterfall':
        shap.plots.waterfall(shap_explanation, show=False)
    else:
        shap.plots.bar(shap_explanation, show=False)
    plt.title("Shap explanation plot", fontsize=16)
    fig.set_figheight(6)
    fig.set_figwidth(9)
    st.pyplot(fig)

    # Display some information
    expander = st.expander("About the SHAP explanation...")
    expander.write("The above plot displays the explanations for the individual prediction of the customer. \
                    It shows the postive and negative contribution of the features. \
                    The final SHAP value is not equal to the prediction probability.")

st.write("")
