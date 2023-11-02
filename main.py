import dataTreatmentUtils
import linearRegPipelineUtils
import pandas as pd
import preprocessing as prep


# Importation de la base de donnée
FILE_PATH = "data\Hotel_Reservations.csv"
DATASET = pd.read_csv(FILE_PATH, sep=',')
target = "booking_status"

# Nettoyage des données : inutile car il n'y a pas de données manquantes dans cette base de données
df = dataTreatmentUtils.removeUselessColumns(DATASET, 30) #(<70% de données sur lignes et colones)


# Prétraitement
# Quantification des données :
df = prep.colToOrdinal(df, ["Booking_ID", "type_of_meal_plan", 
                            "room_type_reserved", "market_segment_type", "booking_status"])

# Séparaison de la base : X les caractéristiques et y la cible (colonne qualité)
X = df.drop(columns=[target])
y = df[target]
print("y : ",y)

# Standardise les données
X = prep.standardize(X)

# Lance la pipeline de la classification

