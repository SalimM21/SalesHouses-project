# SalesHouses-project : Projet de Modélisation Prédictive

## Objectif
Prédire [variable cible] à partir des variables [liste des variables explicatives].

## Installation
1. Cloner le dépôt
2. Créer un environnement virtuel: `python -m venv env`
3. Activer l'environnement
4. Installer les dépendances: `pip install -r requirements.txt`

## Utilisation
Exécuter les notebooks dans l'ordre:
1. `notebooks/exploration.ipynb` - Analyse exploratoire
2. `notebooks/modelisation.ipynb` - Construction des modèles

## Structure du projet

projet-regression/
│
├── data/ # Dossier des données
│ ├── raw/ # Données brutes originales (immutables)
│ ├── interim/ # Données intermédiaires (nettoyées mais non transformées)
│ └── processed/ # Données finales pour modélisation
│ ├── train/ # Jeu d'entraînement
│ └── test/ # Jeu de test
│
├── docs/ # Documentation supplémentaire
│ ├── specifications/ # Cahier des charges
│ └── references/ # Articles, recherches externes
│
├── models/ # Modèles entraînés et sérialisés
│ ├── production/ # Modèles en production
│ └── experimental/ # Modèles expérimentaux
│
├── notebooks/ # Jupyter notebooks
│ ├── 01_EDA.ipynb # Analyse exploratoire
│ ├── 02_Preprocessing.ipynb
│ └── 03_Modeling.ipynb
│
├── reports/ # Résultats générés
│ ├── figures/ # Visualisations sauvegardées
│ ├── final_report.pdf # Rapport final PDF
│ └── presentation.pptx # Support de présentation
│
├── src/ # Code source
│ ├── data/ # Scripts de gestion des données
│ │ ├── make_dataset.py
│ │ └── preprocess.py
│ │
│ ├── features/ # Feature engineering
│ │ ├── build_features.py
│ │ └── feature_selection.py
│ │
│ ├── models/ # Scripts de modélisation
│ │ ├── train_model.py
│ │ ├── predict_model.py
│ │ └── evaluate.py
│ │
│ └── visualization/ # Scripts de visualisation
│ └── visualize.py
│
├── tests/ # Tests unitaires et d'intégration
│ ├── init.py
│ └── test_features.py
│
├── .gitignore # Fichiers à ignorer par Git
├── environment.yml # Environnement Conda
├── requirements.txt # Dépendances pip
├── Makefile # Commandes automatisées
└── README.md # Ce fichier
