# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 01:04:09 2023

@author: Diane
"""

import pandas as pd
import matplotlib.pyplot as plt
import preprocessing as prep
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Import
FILE_PATH = "data/Hotel_Reservations.csv"
df = pd.read_csv(FILE_PATH, sep=',')
colNames = df.columns.tolist()

def canceledPopulation():
    # Group by the number of adults and number of children
    grouped_data = df.groupby(['no_of_adults', 'no_of_children'])['booking_status'].value_counts(normalize=True).unstack().fillna(0)
    
    # Select only the 'Cancelled' column
    cancelled_percentages = grouped_data['Canceled'] * 100
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    cancelled_percentages.plot(kind='bar', color='red')
    
    plt.title('Pourcentage de réservation annulée en fonction de la répartition adultes/enfants')
    plt.xlabel("Nombre d'adultes, Nombre d'enfants")
    plt.ylabel("Pourcentage de réservation annulées")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Booking Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('databaseAnalysisGraphs/cancelPopulation.png')
    plt.show()

def canceledPercentageByDate():
    # Merges the date, month, year columns in a date type column
    df['date_arrival'] = pd.to_datetime(df['arrival_year']*10000 + df['arrival_month']*100 + df['arrival_date'], format='%Y%m%d', errors='coerce')
    df.drop(['arrival_year', 'arrival_month', 'arrival_date'], axis=1, inplace=True)

    grouped_data = df.groupby('date_arrival')['booking_status'].value_counts(normalize=True).unstack().fillna(0) #regroupe les données par date d'arrivée
    
    cancelled_percentages = grouped_data['Canceled'] * 100
    
    # Plot les résultats sous forme de a courbe
    plt.figure(figsize=(15, 8))
    plt.plot(cancelled_percentages.index, cancelled_percentages, marker='o', color='red', linestyle='-', label='Cancelled Percentage')
    
    plt.title("Pourcentage de réservations annulées par date d'arrivée")
    plt.xlabel("Date d'arrivée")
    plt.ylabel("Pourcentage de réservations annulées")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.savefig('databaseAnalysisGraphs/cancelDates.png')
    plt.show()

