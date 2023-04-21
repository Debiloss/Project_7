# -*- coding: utf-8 -*-
""" Librairie personnelle pour manipulation les modèles de machine learning
"""

# ====================================================================

# ====================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
import time
import pickle
import shap
import outils_data
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, \
    explained_variance_score, median_absolute_error
from sklearn.model_selection import cross_validate, RandomizedSearchCV, \
    GridSearchCV, learning_curve  # , cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, \
    BayesianRidge, HuberRegressor, \
    OrthogonalMatchingPursuit, Lars, SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, \
    ExtraTreesRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import eli5
from eli5.sklearn import PermutationImportance
from pprint import pprint
from sklearn import metrics
from sklearn.metrics import confusion_matrix, recall_score, fbeta_score, \
    precision_score, roc_auc_score, average_precision_score
from sklearn.metrics import f1_score, accuracy_score, make_scorer, precision_recall_curve
from sklearn.model_selection import cross_validate

from urllib.parse import urlparse
import mlflow
import mlflow.sklearn



# -----------------------------------------------------------------------
# -- PLOT LES FEATURES IMPORTANCES
# -----------------------------------------------------------------------

def plot_features_importance(features_importance, nom_variables,
                             figsize=(6, 5)):
    '''
    Affiche le liste des variables avec leurs importances par ordre décroissant.
    Parameters
    ----------
    features_importance: les features importances, obligatoire
    nom_variables : nom des variables, obligatoire
    figsize : taille du graphique
    Returns
    -------
    None.
    '''
    df_feat_imp = pd.DataFrame({'feature': nom_variables,
                                'importance': features_importance})
    df_feat_imp_tri = df_feat_imp.sort_values(by='importance')
    
    # BarGraph de visalisation
    plt.figure(figsize=figsize)
    plt.barh(df_feat_imp_tri['feature'], df_feat_imp_tri['importance'])
    plt.yticks(fontsize=20)
    plt.xlabel('Feature Importances (%)')
    plt.ylabel('Variables', fontsize=18)
    plt.title('Comparison des Features Importances', fontsize=30)
    plt.show()
    

def plot_cumultative_feature_importance(df, threshold = 0.9):
    """
    Plots 15 most important features and the cumulative importance of features.
    Prints the number of features needed to reach threshold cumulative importance.
    
    Parameters
    --------
    df : dataframe
        Dataframe of feature importances. Columns must be feature and importance
    threshold : float, default = 0.9
        Threshold for prining information about cumulative importances
        
    Return
    --------
    df : dataframe
        Dataframe ordered by feature importances with a normalized column (sums to 1)
        and a cumulative importance column
    
    """
    
    plt.rcParams['font.size'] = 18
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    # Cumulative importance plot
    plt.figure(figsize = (8, 6))
    plt.plot(list(range(len(df))), df['cumulative_importance'], 'r-')
    plt.xlabel('Number of Features'); plt.ylabel('Cumulative Importance'); 
    plt.title('Cumulative Feature Importance');
    plt.show();
    
    importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
    print('%d variables nécessaires pour %0.2f de cummulative imortance' % (importance_index + 1, threshold))
    
    return df

    
# -----------------------------------------------------------------------
# -- PLOT LES SHAP VALUES
# -----------------------------------------------------------------------


def plot_shape_values(model, x_test):
    '''
    Affiche les SHAPE VALUES.
    Parameters
    ----------
    model: le modèle de machine learning, obligatoire
    x_test :le jeu de test de la matrice X, obligatoire
    Returns
    -------
    None.
    '''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)

    shap.summary_plot(shap_values, x_test, plot_type="bar")

    shap.summary_plot(shap_values, x_test)

    # shap.initjs()
    # shap.force_plot(explainer.expected_value, shap_values[1,:], X_test_log.iloc[1,:])

# -----------------------------------------------------------------------
# -- PLOT LES SHAP VALUES AVEC SKLEARN
# -----------------------------------------------------------------------


def plot_permutation_importance(model, x_test, y_test):
    '''
    Affiche les SHAPE VALUES.
    Parameters
    ----------
    model: le modèle de machine learning, obligatoire
    x_test :le jeu de test de la matrice X, obligatoire
    y_test :le jeu de test de la target, obligatoire
    Returns
    -------
    None.
    '''
    perm_importance = permutation_importance(model, x_test, y_test)

    sorted_idx = perm_importance.importances_mean.argsort()
    plt.figure(figsize=(6, 6))
    plt.barh(x_test.columns[sorted_idx],
             perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance (%)")
    plt.show()
    


# -----------------------------------------------------------------------
# -- PLOT LES SHAP VALUES AVEC ELI5
# -----------------------------------------------------------------------


def plot_permutation_importance_eli5(model, x_test, y_test):
    '''
    Affiche les SHAPE VALUES.
    Parameters
    ----------
    model: le modèle de machine learning, obligatoire
    x_test :le jeu de test de la matrice X, obligatoire
    y_test :le jeu de test de la target, obligatoire
    Returns
    -------
    None.
    '''
    perm = PermutationImportance(model, random_state=21).fit(x_test, y_test)
    display(eli5.show_weights(perm, feature_names=x_test.columns.tolist()))



# -----------------------------------------------------------------------
# -- TRACE LEARNING CURVE POUR VOIR L'OVER ou UNDER FITTING
# -----------------------------------------------------------------------


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Affiche la  learning curve pour le jeu de données de test et d'entraînement
    Parameters
    ----------
    estimator : object qui implemente les méthodes "fit" and "predict"
    title : string, titre du graphique
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training exemples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.grid(False)
    plt.show()



# -----------------------------------------------------------------------
# -- PARTIE 2 : PROJET 7 OPENCLASSROOMS - CLASSIFICATION BINAIRE
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# -- Métrique métier tentant de minimiser le risque d'accord prêt pour la
# -- banque
# -----------------------------------------------------------------------

def custom_score(y_reel, y_pred, taux_tn=1, taux_fp=-5, taux_fn=-20, taux_tp=0):
    '''
    Métrique métier tentant de minimiser le risque d'accord prêt pour la
    banque en pénalisant les faux négatifs.
    Parameters
    ----------
    y_reel : classe réélle, obligatoire (0 ou 1).
    y_pred : classe prédite, obligatoire (0 ou 1).
    taux_tn : Taux de vrais négatifs, optionnel (1 par défaut),
              le prêt est remboursé : la banque gagne de l'argent.
    taux_fp : Taux de faux positifs, optionnel (0 par défaut),
               le prêt est refusé par erreur : la banque perd les intérêts,
               manque à gagner mais ne perd pas réellement d'argent (erreur de
               type I).
    taux_fn : Taux de faux négatifs, optionnel (-10 par défaut),
              le prêt est accordé mais le client fait défaut : la banque perd
              de l'argent (erreur de type II)..
    taux_tp : Taux de vrais positifs, optionnel (1 par défaut),
              Le prêt est refusé à juste titre : la banque ne gagne ni ne perd
              d'argent.
    Returns
    -------
    score : gain normalisé (entre 0 et 1) un score élevé montre une meilleure
            performance
    '''
    # Matrice de Confusion
    (tn, fp, fn, tp) = confusion_matrix(y_reel, y_pred).ravel()
    # Gain total
    gain_tot = tn * taux_tn + fp * taux_fp + fn * taux_fn + tp * taux_tp
    # Gain maximum : toutes les prédictions sont correctes
    gain_max = (fp + tn) * taux_tn + (fn + tp) * taux_tp
    # Gain minimum : on accorde aucun prêt, la banque ne gagne rien
    gain_min = (fp + tn) * taux_fp + (fn + tp) * taux_fn
    
    custom_score = (gain_tot - gain_min) / (gain_max - gain_min)
    
    # Gain normalisé (entre 0 et 1) un score élevé montre une meilleure
    # performance
    return custom_score

def custom_score_2(y_reel, y_pred):
    '''
    Deuxieme métrique métier, plus simpliste que le premier custom score mais ayant toujours pour but de minimiser 
    le nombre de pret accorder à tord.
    Parameters
    ----------
    y_reel : classe réélle, obligatoire (0 ou 1).
    y_pred : classe prédite, obligatoire (0 ou 1).
    Returns
    -------
    score : Un score bas montre une meilleure
            performance, plus le nombre de fn (faux négatifs) est élevé et plus le score est grand.
            On va tenté de trouvé le seuil qui minimise le score métier
    '''
    # Matrice de Confusion
    (tn, fp, fn, tp) = confusion_matrix(y_reel, y_pred).ravel()
    # Gain total
    score = 10*fn + fp
    
    return score
# -----------------------------------------------------------------------
# -- REGLAGE DU SEUIL DE PROBABILITE
# -----------------------------------------------------------------------

def determiner_seuil_probabilite(model, X_valid, y_valid, title, n=1):
    '''
    Déterminer le seuil de probabilité optimal pour la métrique métier.
    Parameters
    ----------
    model : modèle entraîné, obligatoire.
    y_valid : valeur réélle.
    X_valid : données à tester.
    title : titre pour graphique.
    n : gain pour la classe 1 (par défaut) ou 0.
    Returns
    -------
    None.
    '''
    seuils = np.arange(0, 1, 0.01)
    sav_gains = []
 
    for seuil in seuils:

        # Score du modèle : n = 0 ou 1
        y_proba = model.predict_proba(X_valid)[:, n]

        # Score > seuil de solvabilité : retourne 1 sinon 0
        y_pred = (y_proba > seuil)
        y_pred = np.multiply(y_pred, 1)
        
        # Sauvegarde du score de la métrique métier
        sav_gains.append(custom_score(y_valid, y_pred))
    
    df_score = pd.DataFrame({'Seuils' : seuils,
                             'Gains' : sav_gains})
    
    # Score métrique métier maximal
    gain_max = df_score['Gains'].max()
    print(f'Score métrique métier maximal : {gain_max}')
    # Seuil optimal pour notre métrique
    seuil_max = df_score.loc[df_score['Gains'].argmax(), 'Seuils']
    print(f'Seuil maximal : {seuil_max}')
    print('-----------------------------------------------------------')
    print('\n')

    # Affichage du gain en fonction du seuil de solvabilité
    plt.figure(figsize=(12, 6))
    plt.plot(seuils, sav_gains)
    plt.xlabel('Seuil de probabilité')
    plt.ylabel('Métrique métier')
    plt.title(title)
    plt.xticks(np.linspace(0.1, 1, 10))
    
def determiner_seuil_probabilite_2(model, X_valid, y_valid, title, n=1):
    '''
    Déterminer le seuil de probabilité optimal pour la métrique métier.
    Avec un seuil qui minimise 10*fn + fp. Les faux négatifs sont bien plus pénalisé
    Parameters
    ----------
    model : modèle entraîné, obligatoire.
    y_valid : valeur réélle.
    X_valid : données à tester.
    title : titre pour graphique.
    n : gain pour la classe 1 (par défaut) ou 0.
    Returns
    -------
    None.
    '''
    seuils = np.arange(0, 1, 0.01)
    sav_gains = []
    
    
    for seuil in seuils:

        # Score du modèle : n = 0 ou 1
        y_proba = model.predict_proba(X_valid)[:, n]

        # Score > seuil de solvabilité : retourne 1 sinon 0
        y_pred = (y_proba > seuil)
        y_pred = np.multiply(y_pred, 1)
        
        # Sauvegarde du score de la métrique métier
        sav_gains.append(custom_score_2(y_valid, y_pred))
    
    df_score = pd.DataFrame({'Seuils' : seuils,
                             'Gains' : sav_gains})
    
    # Score métrique métier minimal
    gain_min = df_score['Gains'].min()
    print(f'Score métier_2 minimal : {gain_min}')
    # Seuil optimal pour notre métrique
    seuil_min = df_score.loc[df_score['Gains'].argmin(), 'Seuils']
    print(f'Seuil minimal : {seuil_min}')
    print('-----------------------------------------------------------')
    print('\n')

    # Affichage du gain en fonction du seuil de solvabilité
    plt.figure(figsize=(12, 6))
    plt.plot(seuils, sav_gains)
    plt.xlabel('Seuil de probabilité')
    plt.ylabel('Métrique métier 2')
    plt.title(title)
    plt.xticks(np.linspace(0.1, 1, 10))

    
def determiner_seuil_probabilite_F10(model, X_valid, y_valid, title, n=1):
    '''
    Déterminer le seuil de probabilité optimal pour la métrique métier.
    Parameters
    ----------
    model : modèle entraîné, obligatoire.
    y_valid : valeur réélle.
    X_valid : données à tester.
    title : titre pour graphique.
    n : gain pour la classe 1 (par défaut) ou 0.
    Returns
    -------
    None.
    '''
    seuils = np.arange(0, 1, 0.01)
    scores_F10 = []
 
    for seuil in seuils:

        # Score du modèle : n = 0 ou 1
        y_proba = model.predict_proba(X_valid)[:, n]

        # Score > seuil de solvabilité : retourne 1 sinon 0
        y_pred = (y_proba > seuil)
        y_pred = np.multiply(y_pred, 1)
        
        # Sauvegarde du score de la métrique métier
        scores_F10.append(fbeta_score(y_valid, y_pred, beta=10))
    
    df_score = pd.DataFrame({'Seuils' : seuils,
                             'Gains' : scores_F10})
    print(title)
    # Score métrique métier maximal
    gain_max = df_score['Gains'].max()
    print(f'Score F10 maximal : {gain_max}')
    # Seuil optimal pour notre métrique
    seuil_max = df_score.loc[df_score['Gains'].argmax(), 'Seuils']
    print(f'Seuil maximal : {seuil_max}')
    print('-----------------------------------------------------------')
    print('\n')

    # Affichage du gain en fonction du seuil de solvabilité
    plt.figure(figsize=(12, 6))
    plt.plot(seuils, scores_F10)
    plt.xlabel('Seuil de probabilité')
    plt.ylabel('Score F10')
    plt.title(title)
    plt.xticks(np.linspace(0.1, 1, 10))
    
# ------------------------------------------------------------------------
# -- ENTRAINER/PREDIRE/CALCULER SCORES -  MODELE DE CLASSIFICATION BINAIRE
# ------------------------------------------------------------------------


def process_classification(model, X_train, X_valid, y_train, y_valid,
                           df_resultats, titre, result_table, recall_table, affiche_res=True,
                           affiche_matrice_confusion=True, roc_cur=True, avg_precision=True):
    """
    Lance un modele de classification binaire, effectue cross-validation
    et sauvegarde des scores.
    Parameters
    ----------
    model : modèle de lassification initialisé, obligatoire.
    X_train : train set matrice X, obligatoire.
    X_valid : test set matrice X, obligatoire.
    y_train : train set vecteur y, obligatoire.
    y_valid : test set, vecteur y, obligatoire.
    df_resultats : dataframe sauvegardant les scores, obligatoire
    titre : titre à inscrire dans le tableau de sauvegarde, obligatoire.
    affiche_res : affiche le tableau de résultat (optionnel, True par défaut).
    Returns
    -------
    df_resultats : Le dataframe de sauvegarde des performances.
    y_pred : Les prédictions pour le modèle
    """
    # Top début d'exécution
    time_start = time.time()
    
    # Sauvegarde du modèle de classification entraîné
    with open(r'C:\Users\Sofia\OneDrive\Documents\OpenClassrooms\Project_7\modelisation\modele_' + titre + '.pickle', 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    
    # Entraînement du modèle avec le jeu d'entraînement du jeu d'entrainement
    model.fit(X_train, y_train)
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # Sauvegarde du modèle de classification entraîné
    with mlflow.start_run():
    
        # Top fin d'exécution
        time_end_train = time.time()

        # Prédictions avec le jeu de validation du jeu d'entraînement
        y_pred = model.predict(X_valid)

        # Top fin d'exécution
        time_end = time.time()

        # Probabilités
        y_proba = model.predict_proba(X_valid)[:, 1]

        # Calcul des métriques
        # Rappel/recall sensibilité
        recall = recall_score(y_valid, y_pred)
        # Précision
        precision = precision_score(y_valid, y_pred)
        # F-mesure ou Fbeta
        f1_score = fbeta_score(y_valid, y_pred, beta=1)
        f5_score = fbeta_score(y_valid, y_pred, beta=5)
        f10_score = fbeta_score(y_valid, y_pred, beta=10)
        # Score ROC AUC aire sous la courbe ROC
        roc_auc = roc_auc_score(y_valid, y_proba)
        # Score PR AUC aire sous la courbe précion/rappel
        pr_auc = average_precision_score(y_valid, y_proba)
        # Métrique métier
        banque_score = custom_score(y_valid, y_pred)
        banque_score_2 = custom_score_2(y_valid, y_pred)

        table_roc = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])
        fpr, tpr, _ = metrics.roc_curve(y_valid, y_proba)
        table_roc = table_roc.append({'classifiers':model.__class__.__name__,
                                            'fpr':fpr, 
                                            'tpr':tpr, 
                                            'auc':roc_auc,}, ignore_index=True)

        table_roc.set_index('classifiers', inplace=True)

        table_rec = pd.DataFrame(columns=['classifiers', 'precision','recall','threshold', 'score'])    
        prec, rec, threshold = precision_recall_curve(y_valid, y_proba)
        avg_score = average_precision_score(y_valid, y_proba)

        table_rec = table_rec.append({'classifiers':model.__class__.__name__,
                                            'precision':prec, 
                                            'recall':rec, 
                                            'threshold':threshold,
                                            'score' :avg_score}, ignore_index=True)

        table_rec.set_index('classifiers', inplace=True)


        # durée d'exécution d'entraînement
        time_exec_train = time_end_train - time_start
        # durée d'exécution entraînement + validation
        time_execution = time_end - time_start

        # cross validation
        scoring = ['roc_auc', 'recall', 'precision']
        scores = cross_validate(model, X_train, y_train, cv=10,
                                scoring=scoring, return_train_score=True)

        # Sauvegarde des performances
        df_resultats = df_resultats.append(pd.DataFrame({
            'Modèle': [titre],
            'Rappel': [recall],
            'Précision': [precision],
            'F1': [f1_score],
            'F5': [f5_score],
            'F10': [f10_score],
            'ROC_AUC': [roc_auc],
            'PR_AUC': [pr_auc],
            'Metier_score': [banque_score],
            'Metier_score_2': [banque_score_2],
            'Durée_train': [time_exec_train],
            'Durée_tot': [time_execution],
            # Cross-validation
            'Train_roc_auc_CV': [scores['train_roc_auc'].mean()],
            'Train_roc_auc_CV +/-': [scores['train_roc_auc'].std()],
            'Test_roc_auc_CV': [scores['test_roc_auc'].mean()],
            'Test_roc_auc_CV +/-': [scores['test_roc_auc'].std()],
            'Train_recall_CV': [scores['train_recall'].mean()],
            'Train_recall_CV +/-': [scores['train_recall'].std()],
            'Test_recall_CV': [scores['test_recall'].mean()],
            'Test_recall_CV +/-': [scores['test_recall'].std()],
            'Train_precision_CV': [scores['train_precision'].mean()],
            'Train_precision_CV +/-': [scores['train_precision'].std()],
            'Test_precision_CV': [scores['test_precision'].mean()],
            'Test_precision_CV +/-': [scores['test_precision'].std()],
        }), ignore_index=True)

        result_table = result_table.append(table_roc)
        recall_table = recall_table.append(table_rec)
        
        mlflow.log_metric("ROC_AUC", roc_auc)
        mlflow.log_metric("PR_AUC", pr_auc)
        mlflow.log_metric("Rappel", recall)
        mlflow.log_metric("Précision", precision)
        mlflow.log_metric("F1", f1_score)
        mlflow.log_metric("F5", f5_score)
        mlflow.log_metric("F10", f10_score)
        mlflow.log_metric("Durée_tot", time_execution)
        mlflow.log_metric("Métier_score", banque_score)
        mlflow.log_metric("Métier_score_2", banque_score_2)
        
        mlflow.sklearn.log_model(model, "model", registered_model_name=titre)
    
    mlflow.end_run()
    
    # Sauvegarde du tableau de résultat
    with open(r'C:\Users\Sofia\OneDrive\Documents\OpenClassrooms\Project_7\modelisation\df_resultats.pickle', 'wb') as df:
        pickle.dump(df_resultats, df, pickle.HIGHEST_PROTOCOL)
    
    if affiche_res:
        mask = df_resultats['Modèle'] == titre
        display(df_resultats[mask].style.hide_index())

    if affiche_matrice_confusion:
        afficher_matrice_confusion(y_valid, y_pred, titre)

    return df_resultats, result_table, recall_table
    

def process_classification_seuil(model, seuil, X_train, X_valid, y_train,
                                 y_valid, df_res_seuil, titre,
                                 affiche_res=True,
                                 affiche_matrice_confusion=True):
    """
    Lance un modele de classification binaire, effectue cross-validation
    et sauvegarde des scores.
    Parameters
    ----------
    model : modèle de lassification initialisé, obligatoire.
    seuil : seuil de probabilité optimal.
    X_train : train set matrice X, obligatoire.
    X_valid : test set matrice X, obligatoire.
    y_train : train set vecteur y, obligatoire.
    y_valid : test set, vecteur y, obligatoire.
    df_res_seuil : dataframe sauvegardant les scores, obligatoire
    titre : titre à inscrire dans le tableau de sauvegarde, obligatoire.
    affiche_res : affiche le tableau de résultat (optionnel, True par défaut).
    Returns
    -------
    df_resultats : Le dataframe de sauvegarde des performances.
    y_pred : Les prédictions pour le modèle
    """
    # Top début d'exécution
    time_start = time.time()

    # Entraînement du modèle avec le jeu d'entraînement du jeu d'entrainement
    model.fit(X_train, y_train)
    
    # Sauvegarde du modèle de classification entraîné
    with open(r'C:\Users\Sofia\OneDrive\Documents\OpenClassrooms\Project_7\modelisation\modele_' + titre + '.pickle', 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # Sauvegarde du modèle de classification entraîné
    with mlflow.start_run():
    
        # Top fin d'exécution
        time_end_train = time.time()

        # Score du modèle : n = 0 ou 1
        # Probabilités
        y_proba = model.predict_proba(X_valid)[:, 1]

        # Prédictions avec le jeu de validation du jeu d'entraînement
        # Score > seuil de probabilité : retourne 1 sinon 0
        y_pred = (y_proba > seuil)
        y_pred = np.multiply(y_pred, 1)

        # Top fin d'exécution
        time_end = time.time()

        # Calcul des métriques
        # Rappel/recall sensibilité
        recall = recall_score(y_valid, y_pred)
        # Précision
        precision = precision_score(y_valid, y_pred)
        # F-mesure ou Fbeta
        f1_score = fbeta_score(y_valid, y_pred, beta=1)
        f5_score = fbeta_score(y_valid, y_pred, beta=5)
        f10_score = fbeta_score(y_valid, y_pred, beta=10)
        # Score ROC AUC aire sous la courbe ROC
        roc_auc = roc_auc_score(y_valid, y_proba)
        # Score PR AUC aire sous la courbe précion/rappel
        pr_auc = average_precision_score(y_valid, y_proba)
        # Métrique métier
        banque_score = custom_score(y_valid, y_pred)
        banque_score_2 = custom_score_2(y_valid, y_pred)
        
        # durée d'exécution d'entraînement
        time_exec_train = time_end_train - time_start
        # durée d'exécution entraînement + validation
        time_execution = time_end - time_start

        # cross validation
        scoring = ['roc_auc', 'recall', 'precision']
        scores = cross_validate(model, X_train, y_train, cv=10,
                                scoring=scoring, return_train_score=True)

        # Sauvegarde des performances
        df_res_seuil = df_res_seuil.append(pd.DataFrame({
            'Modèle': [titre],
            'Rappel': [recall],
            'Précision': [precision],
            'F1': [f1_score],
            'F5': [f5_score],
            'F10': [f10_score],
            'ROC_AUC': [roc_auc],
            'PR_AUC': [pr_auc],
            'Metier_score': [banque_score],
            'Metier_score_2': [banque_score_2],
            'Durée_train': [time_exec_train],
            'Durée_tot': [time_execution],
            # Cross-validation
            'Train_roc_auc_CV': [scores['train_roc_auc'].mean()],
            'Train_roc_auc_CV +/-': [scores['train_roc_auc'].std()],
            'Test_roc_auc_CV': [scores['test_roc_auc'].mean()],
            'Test_roc_auc_CV +/-': [scores['test_roc_auc'].std()],
            'Train_recall_CV': [scores['train_recall'].mean()],
            'Train_recall_CV +/-': [scores['train_recall'].std()],
            'Test_recall_CV': [scores['test_recall'].mean()],
            'Test_recall_CV +/-': [scores['test_recall'].std()],
            'Train_precision_CV': [scores['train_precision'].mean()],
            'Train_precision_CV +/-': [scores['train_precision'].std()],
            'Test_precision_CV': [scores['test_precision'].mean()],
            'Test_precision_CV +/-': [scores['test_precision'].std()],
        }), ignore_index=True)

    # Sauvegarde du tableau de résultat
        mlflow.log_metric("ROC_AUC", roc_auc)
        mlflow.log_metric("PR_AUC", pr_auc)
        mlflow.log_metric("Rappel", recall)
        mlflow.log_metric("Précision", precision)
        mlflow.log_metric("F1", f1_score)
        mlflow.log_metric("F5", f5_score)
        mlflow.log_metric("F10", f10_score)
        mlflow.log_metric("Durée_tot", time_execution)
        mlflow.log_metric("Métier_score", banque_score)
        mlflow.log_metric("Métier_score_2", banque_score_2)
        
        mlflow.sklearn.log_model(model, "model", registered_model_name=titre)
      
    mlflow.end_run()
    
    # Sauvegarde du tableau de résultat
    with open(r'C:\Users\Sofia\OneDrive\Documents\OpenClassrooms\Project_7\modelisation\df_res_seuil.pickle', 'wb') as df:
        pickle.dump(df_res_seuil, df, pickle.HIGHEST_PROTOCOL)
    
    if affiche_res:
        mask = df_res_seuil['Modèle'] == titre
        display(df_res_seuil[mask].style.hide_index())

    if affiche_matrice_confusion:
        afficher_matrice_confusion(y_valid, y_pred, titre)

    return df_res_seuil

# ------------------------------------------------------------------------
# -- SAUVEGARDE DES TAUX
# -- TN : vrais négatifs, TP : vrais positifs
# -- FP : faux positifs, FN : faux négatifs
# ------------------------------------------------------------------------

def sauvegarder_taux(titre_modele, FN, FP, TP, TN, df_taux):
    """
    Lance un modele de classification binaire, effectue cross-validation
    et sauvegarde des scores.
    Parameters
    ----------
    model : modèle de lassification initialisé, obligatoire.
    FN : nombre de faux négatifs, obligatoire.
    FP : nombre de faux positifs, obligatoire.
    TN : train set vecteur y, obligatoire.
    TP : test set, vecteur y, obligatoire.
    df_taux : dataframe sauvegardant les taux, obligatoire
    titre : titre à inscrire dans le tableau de sauvegarde, obligatoire.
    Returns
    -------
    df_taux : Le dataframe de sauvegarde des taux.
    """

    # Sauvegarde des performances
    df_taux = df_taux.append(pd.DataFrame({
        'Modèle': [titre_modele],
        'FN': [FN],
        'FP': [FP],
        'TP': [TP],
        'TN': [TN]
    }), ignore_index=True)

    # Sauvegarde du tableau de résultat
    with open(r'C:\Users\Sofia\OneDrive\Documents\OpenClassrooms\Project_7\modelisation\df_taux.pickle', 'wb') as df:
        pickle.dump(df_taux, df, pickle.HIGHEST_PROTOCOL)
    
    return df_taux



# -----------------------------------------------------------------------
# -- MATRICE DE CONFUSION DE LA CLASSIFICATION BINAIRE
# -----------------------------------------------------------------------

def afficher_matrice_confusion(y_true, y_pred, title):

    plt.figure(figsize=(6, 4))

    cm = confusion_matrix(y_true, y_pred)
    
    labels = ['Non défaillants', 'Défaillants']
    
    sns.heatmap(cm,
                xticklabels=labels,
                yticklabels=labels,
                annot=True,
                fmt='d',
                cmap=plt.cm.Blues)
    plt.title(f'Matrice de confusion de : {title}')
    plt.ylabel('Classe réelle')
    plt.xlabel('Classe prédite')
    plt.show()    

def roc_plot(result_table):
    
    fig = plt.figure(figsize=(10,8))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'], 
                 result_table.loc[i]['tpr'], 
                 label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc'])
                 )

    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')

    plt.show()
    
    
def avg_plot(recall_table):
    
    fig = plt.figure(figsize=(10,8))

    for i in recall_table.index:
        plt.plot(recall_table.loc[i]['recall'], 
                 recall_table.loc[i]['precision'], 
                 label="{}, average_precision_score={:.3f}".format(i, recall_table.loc[i]['score'])
                 )

    #plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Recall", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("Precision", fontsize=15)

    plt.title('Precision-Recall Curve', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='upper right')

    plt.show()

