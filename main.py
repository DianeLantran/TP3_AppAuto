import pandas as pd
import preprocessing as prep
import classification
import src.plotClassifiers as pltModels

# Importation de la base de donnée
FILE_PATH = "data\Hotel_Reservations.csv"
df = pd.read_csv(FILE_PATH, sep=',')
target = "booking_status"

# Nettoyage des données : inutile car il n'y a pas de données manquantes dans cette base de données

# Prétraitement
#fusion des colonnes jour/mois/annee d'arrivée en une colonne
# df['date_arrival'] = pd.to_datetime(df['arrival_year']*10000 + df['arrival_month']*100 + df['arrival_date'], format='%Y%m%d', errors='coerce')
# df.drop(['arrival_year', 'arrival_month', 'arrival_date'], axis=1, inplace=True)

# Quantification des données :
categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', "booking_status"]
df = prep.colToOrdinal(df, categorical_cols)

# Séparaison de la base : X les caractéristiques et y la cible (colonne qualité)
features = df.columns.difference(['Booking_ID', 'booking_status'])
X = df[features]
y = df[target]


# Standardise les données
X = prep.standardize(X)

# Lance la pipeline de la classification
trained_models = classification.classify(X, y)
pltModels.plotModels(trained_models, X, y)