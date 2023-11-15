import pandas as pd
import preprocessing as prep
import classification
import plotClassifiers as pltModels
import evaluationUtils as eva

# Importation de la base de donnée
FILE_PATH = "data\Hotel_Reservations.csv"
df = pd.read_csv(FILE_PATH, sep=',')
target = "booking_status"

# Nettoyage des données : inutile car il n'y a pas de données manquantes dans cette base de données

# Encodage des données :
categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', "booking_status"]
df = prep.colToOrdinal(df, categorical_cols)

# Séparaison de la base : X les caractéristiques et y la cible (colonne qualité)
features = df.columns.difference(['Booking_ID', 'booking_status'])
X = df[features]
y = df[target]


# Standardise les données
X = prep.standardize(X)

# Lance la pipeline de la classification
trained_models, resultsList = classification.classify(X, y)

#eva.graphScores(resultsList)
eva.ROCAndAUC(resultsList)