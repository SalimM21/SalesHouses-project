import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np




df = pd.read_csv('appartements_data_db.csv')
print(df.info())
df.describe()

# Identifier les valeurs manquantes et les doublons
df.isnull().sum()
df.duplicated().sum()

# Statistiques
print(df['price'].describe())

# Visualisation
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.histplot(df['price'], kde=True, bins=30)
plt.title('Histogramme du price')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['price'])
plt.title('Boxplot du price')

plt.tight_layout()
plt.show()

# Prétraitement des données 
# 1. Extraction des Équipements (equipment) en Colonnes Booléennes
df_equipment = df['equipment'].str.get_dummies(sep=',') # permet de transformer rapidement une variable catégorielle en variables dummy
df = pd.concat([df, df_equipment], axis=1)
df.drop('equipment', axis=1, inplace=True)

# 2. Conversion de price (type Objet → Float)
df['price'] = df['price'].str.replace(r'[^\d]', '', regex=True)  # Effectue un remplacement utilisant une expression régulière
df['price'] = df['price'].astype(float)
# Vérification
print(df['price'].dtype)  # Doit afficher "float64"

# 3. Suppression des Colonnes Inutiles (equipment, link)
df.drop(['link'], axis=1, inplace=True)

# 4. Traitement de city_name (Uniformisation des Noms de Villes)
# afficher la liste des noms de ville dans une colonne du DataFrame
df['ville'] = df['city_name'].str.split(',').str[0] # divise chaque chaîne à chaque virgule (,)
print(df['ville'])

# supprimer les villes en double et ignorer les valeurs nan
ville = [v for v in set(df['ville']) if pd.notnull(v)]
ville.sort()
print(ville)

# Remplacer les noms arabes par leur équivalent français
city_mapping = {
    "أكادير" : "Agadir",
    "الرباط" : "Rabat",
    "القنيطرة" :"kinetra",
    "المحمدية" : "Mouhmadia",
    "الدار البيضاء": "Casablanca",
    "مراكش": "Marrakech",
    "طنجة": "Tanger",
}

df['city_name'] = df['city_name'].replace(city_mapping)

# Remplacer les valeurs manquantes par "Unknown"
df['city_name'] = df['city_name'].fillna("Unknown", inplace=True)

print(df['city_name'].unique())  # Affiche les villes uniformisées

# Gestion des valeurs manquantes
# - Pour les colonnes numériques : imputer les valeurs manquantes par la médiane.
# - Pour les colonnes catégorielles (chaînes de caractères) : imputer avec "Unknown".

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns # Sélectionner uniquement les colonnes numériques
imputer_median = SimpleImputer(strategy='median')
df[numeric_cols] = imputer_median.fit_transform(df[numeric_cols])

# 2. Colonnes Catégorielles (Strings) → Imputation par "Unknown"
categorical_cols = df.select_dtypes(include=['object', 'category']).columns # Sélectionner les colonnes catégorielles
df[categorical_cols] = df[categorical_cols].fillna("Unknown") # Remplacer les NaN par "Unknown"

# Détection et suppression des valeurs aberrantes:
# - Utiliser des méthodes statistiques (boîtes à moustaches, z-score, IQR) pour détecter les outliers.
# - Supprimer les lignes contenant des valeurs aberrantes sur des colonnes clés (ex: price, surface_area, etc.).

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df_clean = df.copy()
for col in ['price', 'surface_area']:  # Colonnes à vérifier
    df_clean = remove_outliers_iqr(df_clean, col)

print(f"Taille avant/après : {len(df)} → {len(df_clean)} lignes")

# Encodage des variables catégorielles: --> Appliquer un Label Encoding selon le modèle utilisé, en particulier sur city_name.
label_encoder = LabelEncoder() # Initialisation
df['city_name_encoded'] = label_encoder.fit_transform(df['city_name']) # Application sur 'city_name'
print(df[['city_name', 'city_name_encoded']].head())

# Mise à l’échelle des variables: --> Appliquer une normalisation (MinMaxScaler) ou une standardisation (StandardScaler) sur les variables numériques pour harmoniser les échelles.

# Sélectionner uniquement les colonnes numériques
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
print("Colonnes numériques :", list(numeric_cols))

# Initialisation + application
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Vérification (moyenne ~0, écart-type ~1)
print(df[numeric_cols].describe())

# Colonnes à normaliser (ex: [0, 1])
scaler = MinMaxScaler(feature_range=(0, 1)) 
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Vérification (valeurs entre 0 et 1)
print(df[numeric_cols].describe())

# Sélection des variables explicatives:
# - Choisir les variables numériques corrélées au prix (corr > 0.15).
# - Vérifier que les variables choisies ne sont pas fortement corrélées entre elles pour éviter la redondance.


corr_matrix = df.corr(numeric_only=True) # Calcul de la matrice de corrélation
price_corr = corr_matrix['price'].sort_values(ascending=False) # Extraction des corrélations avec 'price'
print("Corrélation avec le prix :\n", price_corr)
selected_vars = price_corr[abs(price_corr) > 0.15].index.tolist() # Sélection des variables avec |corr| > 0.15
selected_vars.remove('price')  # Exclure la cible elle-même
print("\nVariables sélectionnées :", selected_vars)

# Séparation des données:
# - Définir la variable cible y = df["price"].
# - Définir les variables explicatives X à partir des colonnes sélectionnées. 
# - Diviser les données en ensemble d’entraînement et de test (80% / 20%) avec train_test_split.


y = df["price"] # Variable cible
# Variables explicatives (colonnes sélectionnées précédemment)
selected_features = ['Balcon/Chauffage/Climatisation/Cuisine Équipée/Parking/Sécurité/Terrasse', 'nb_rooms', 'salon', 'Balcon/Chauffage/Climatisation/Cuisine Équipée/Meublé/Parking/Sécurité/Terrasse']
X = df[selected_features]

# Vérification
print("Dimensions de X :", X.shape)
print("Dimensions de y :", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42 # Séparation aléatoire (avec random_state pour la reproductibilité)
)

# Vérification des tailles
print("Train set :", X_train.shape, y_train.shape)
print("Test set  :", X_test.shape, y_test.shape)

# Entraîner plusieurs modèles :
# - Régression Linéaire
# - Random Forest Regressor
# - SVR (Support Vector Regressor)
# - Gradient Boosting Regressor

# (1) Régression Linéaire

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Linear Regression:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_lr):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_lr):.4f}")
print(f"R²: {r2_score(y_test, y_pred_lr):.4f}")

# (2) Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)  # Pas besoin de scaling pour les arbres
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_rf):.2f}")
print(f"R²: {r2_score(y_test, y_pred_rf):.2f}")

# (3) Support Vector Regressor (SVR)
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X_train, y_train)  # Scaling crucial pour SVR
y_pred_svr = svr.predict(X_test)

print("\nSVR:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_svr):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_svr):.2f}")
print(f"R²: {r2_score(y_test, y_pred_svr):.2f}")

# (4) Gradient Boosting Regressor (GBR)
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gbr.fit(X_train, y_train)  # Pas besoin de scaling
y_pred_gbr = gbr.predict(X_test)

print("\nGradient Boosting:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_gbr):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_gbr):.2f}")
print(f"R²: {r2_score(y_test, y_pred_gbr):.2f}")

# 3. Comparaison des modèles
results = {
    'Linear Regression': [y_pred_lr],
    'Random Forest': [y_pred_rf],
    'SVR': [y_pred_svr],
    'Gradient Boosting': [y_pred_gbr]
}

for name, pred in results.items():
    print(f"\n{name}:")
    print(f"MAE: {mean_absolute_error(y_test, pred[0]):.2f}")
    print(f"MSE: {mean_squared_error(y_test, pred[0]):.2f}")
    print(f"R²: {r2_score(y_test, pred[0]):.2f}")

# Validation croisée -->Utiliser la validation croisée (cross-validation) pour évaluer la robustesse des modèles sur différentes portions du jeu de données.

# 1. Création du pipeline (standardisation + SVR)
svr_pipeline = make_pipeline(
    StandardScaler(),  # Crucial pour SVR
    SVR(kernel='rbf', C=10, epsilon=0.1)  # Paramètres initiaux
)

# 2. Configuration de la validation croisée
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5 folds aléatoires

# 3. Métriques à évaluer (R², MAE, MSE)
scoring = {
    'R2': 'r2',
    'MAE': 'neg_mean_absolute_error',
    'MSE': 'neg_mean_squared_error'
}

# 4. Exécution de la validation croisée
cv_results = cross_validate(
    svr_pipeline,
    X,  # Vos features
    y,  # Votre target
    cv=kf,
    scoring=scoring,
    return_train_score=True  # Pour détecter overfitting
)

# 5. Affichage des résultats
print("Résultats moyens par fold (Test):")
print(f"R²: {np.mean(cv_results['test_R2']):.2f} (±{np.std(cv_results['test_R2']):.2f})")
print(f"MAE: {-np.mean(cv_results['test_MAE']):.2f} (±{np.std(cv_results['test_MAE']):.2f})")
print(f"MSE: {-np.mean(cv_results['test_MSE']):.2f} (±{np.std(cv_results['test_MSE']):.2f})")

print("\nComparaison train/test (Overfitting check):")
print(f"Différence R² train-test: {np.mean(cv_results['train_R2']) - np.mean(cv_results['test_R2']):.2f}")

# Optimisation des hyperparamètres -->Utiliser GridSearchCV ou RandomizedSearchCV pour rechercher les meilleurs hyperparamètres pour chaque modèle.
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

# Distributions pour tirage aléatoire
param_dist = {
    'svr__C': loguniform(1e-3, 1e3),  # Distribution logarithmique
    'svr__epsilon': loguniform(0.01, 0.5),
    'svr__kernel': ['rbf', 'linear'],
    'svr__gamma': loguniform(1e-3, 1e1)
}

random_search = RandomizedSearchCV(
    svr_pipeline,
    param_distributions=param_dist,
    n_iter=50,  # Nombre d'itérations aléatoires
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)

# Sauvegarder le modèle entraîné (model.pkl)

import joblib
joblib.dump(best_model, 'meilleur_modele_regression.pkl')