- Dossier Dashboard / API

o Main.py: code de l�API
o Scoring_dashboard.py: code du dashboard
o Main_test.py : code des tests unitaires (Pytest)
o Best_model.pickle�: Mod�le utilis� pour pr�diction
o lgbm_shap_explainer.pickle�: interpr�tation globale des variables du mod�le
o data_drift_report.html: rapport datdrift
o data_model_test, new_test, test_info .csv: dataset 
o Procfile: fichier de configuration des diff�rents processus de d�ploiement 
o Dockerfile�: paquet contenant les d�pendances et configurations n�cessaires � l�application
o Requirements�: fichier avec les versions � installer pour les diff�rents packages
o Runtime�: fichier pr�cisant la version python � utilis�
o .github\workflows\build_deploy�: fichier de configuration pour un syst�me de gestion de d�ploiement continu


- Dossier Notebook

	EDA_1 / EDA_2 : Notebook d'analyse exploratoire (https://www.kaggle.com/code/arezoodahesh/home-credit-	default-risk-part01-eda#Data-Understanding)

	Preprocess : preprocessing des data sets train&test

	Mod�lisation part 1 & 2 : Mod�lisation pour notre probl�matique, resultats, interpr�tabilit�

	Analyse data : Analyse des pr�dictions sur dataset test et datadrift

	Prep data : Pr�paration des donn�es pour un chargement optimale pour le dashboard

	data_visu : diff�rentes fonctions pour la visualisation des donn�es

	outils_data : diff�rentes fonctions pour modifi� la data

	outils_model: diff�rentes fonctions pour la mod�lisation