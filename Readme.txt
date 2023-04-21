- Dossier Dashboard / API

o Main.py: code de l’API
o Scoring_dashboard.py: code du dashboard
o Main_test.py : code des tests unitaires (Pytest)
o Best_model.pickle : Modèle utilisé pour prédiction
o lgbm_shap_explainer.pickle : interprétation globale des variables du modèle
o data_drift_report.html: rapport datdrift
o data_model_test, new_test, test_info .csv: dataset 
o Procfile: fichier de configuration des différents processus de déploiement 
o Dockerfile : paquet contenant les dépendances et configurations nécessaires à l’application
o Requirements : fichier avec les versions à installer pour les différents packages
o Runtime : fichier précisant la version python à utilisé
o .github\workflows\build_deploy : fichier de configuration pour un système de gestion de déploiement continu


- Dossier Notebook

	EDA_1 / EDA_2 : Notebook d'analyse exploratoire (https://www.kaggle.com/code/arezoodahesh/home-credit-	default-risk-part01-eda#Data-Understanding)

	Preprocess : preprocessing des data sets train&test

	Modélisation part 1 & 2 : Modélisation pour notre problématique, resultats, interprétabilité

	Analyse data : Analyse des prédictions sur dataset test et datadrift

	Prep data : Préparation des données pour un chargement optimale pour le dashboard

	data_visu : différentes fonctions pour la visualisation des données

	outils_data : différentes fonctions pour modifié la data

	outils_model: différentes fonctions pour la modélisation