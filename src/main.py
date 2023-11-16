import pandas as pd
import preprocessing as prep
import classification
import evaluationUtils as eva

# Importation de la base de donnée
FILE_PATH = "../data/Hotel_Reservations.csv"
df = pd.read_csv(FILE_PATH, sep=',')

# Nettoyage des données : inutile car il n'y a pas de données manquantes dans 
# cette base de données

target = "booking_status"
features = df.columns.difference(['Booking_ID', 'booking_status'])
categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', "booking_status"]

# Preprocessing
X, y = prep.preprocess(df, categorical_cols, features, target)

# Lance la pipeline de la classification
trained_models, resultsList = classification.classify(X, y)

#eva.graphScores(resultsList)
eva.ROCAndAUC(resultsList)