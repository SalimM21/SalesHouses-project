# SalesHouses-project : Projet de Modélisation Prédictive

## Objectif
L’entreprise souhaite moderniser son offre en lançant un simulateur intelligent d’évaluation immobilière . Cet outil permettra aux utilisateurs de saisir les caractéristiques clés d’un bien immobilier (superficie, nombre de chambres, localisation, état, etc.) et d’obtenir une estimation précise du prix de marché, adapté au contexte marocain.

## Fonctionnalités principales

- Analyse exploratoire des données (EDA)
- Prétraitement des données (nettoyage, encodage, normalisation)
- Entraînement de plusieurs modèles (Linear Regression, Random Forest, SVR, Gradient Boosting)
- Évaluation avec MAE, RMSE, R²
- Sélection et sauvegarde du meilleur modèle
- Intégration prévue dans une future interface web
- 
---

## Méthodologie

Le projet suit une démarche rigoureuse en plusieurs étapes :

1. **Chargement et exploration des données** (`pandas`)
2. **Analyse exploratoire (EDA)** : compréhension de la structure, visualisations, corrélations
3. **Nettoyage et prétraitement** :
   - Conversion de la variable `price` en float
   - Encodage des équipements (get_dummies)
   - Traitement de la colonne `city_name`
   - Gestion des valeurs manquantes
   - Suppression des outliers (Boites à moustaches)
   - ...
4. **Sélection des variables explicatives** : basées sur la corrélation avec la cible
5. **Mise à l’échelle des variables numériques** (StandardScaler)
6. **Entraînement de plusieurs modèles supervisés** :
   - Régression linéaire
   - Random Forest Regressor
   - SVR (Support Vector Regressor)
   - Gradient Boosting Regressor
7. **Évaluation des performances** :
   - Métriques utilisées : R², RMSE, MAE, MSE
   - Validation croisée (`cross_val_score`)
8. **Optimisation des hyperparamètres** (`GridSearchCV`)
9. **Sauvegarde du meilleur modèle** (`joblib`)
10. **Déploiement d'une interface utilisateur avec Streamlit**
## Utilisation
Exécuter les notebooks dans l'ordre:
1. `notebooks/main.ipynb` - Analyse exploratoire et Construction des modèles

## Structure du dossier

```
Brief_2_SalesHouses/
├── appartements_data_db.csv
├── meilleur_modele_regression.pkl
├── README.md
├── main.py
└── main.ipynb
```

## Prérequis

- Python 3.x
- Les bibliothèques suivantes :
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - jupyter

##  Application Streamlit

L’utilisateur peut accéder à une application interactive qui lui permet de :
- Saisir les caractéristiques du bien (surface, nb pièces, équipements, etc.)
- Obtenir instantanément une estimation du prix

Pour lancer l'application localement :

```bash
pip install -r requirements.txt
streamlit run app.py
```
