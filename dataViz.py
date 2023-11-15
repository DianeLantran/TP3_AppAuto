# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 01:04:09 2023

@author: Diane Lantran
"""

import pandas as pd
import matplotlib.pyplot as plt
import preprocessing as prep
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Import
df = pd.read_csv("data/Hotel_Reservations.csv", sep=',')
colNames = df.columns.tolist()
nb_reservation = df.shape[0]

def canceledPopulation():
    # Group by the number of adults and number of children
    grouped_data = df.groupby(['no_of_adults', 'no_of_children'])['booking_status'].value_counts(normalize=True).unstack().fillna(0)
    
    # Select only the 'Cancelled' column
    cancelled_amount = grouped_data['Canceled']*100 #percentage of cancellation for each group
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    cancelled_amount.plot(kind='bar', color='red')
    
    plt.title('Pourcentage de réservation annulée en fonction de la répartition adultes/enfants')
    plt.xlabel("Nombre d'adultes, Nombre d'enfants")
    plt.ylabel("Pourcentage de réservation annulées")
    plt.xticks(rotation=45, ha='right')
    plt.savefig('databaseAnalysisGraphs/cancelPopulation.png')
    plt.show()

def canceledPercentageByDate():
    # Merges the date, month, year columns in a date type column
    if 'date_arrival' not in df.columns:
        df['date_arrival'] = pd.to_datetime(df['arrival_year']*10000 + df['arrival_month']*100 + df['arrival_date'], format='%Y%m%d', errors='coerce')
        df.drop(['arrival_year', 'arrival_month', 'arrival_date'], axis=1, inplace=True)

    grouped_data = df.groupby('date_arrival')['booking_status'].value_counts(normalize=True).unstack().fillna(0) #regroupe les données par date d'arrivée
    
    cancelled_percentages = grouped_data['Canceled'] * 100/(grouped_data['Canceled']+grouped_data['Not_Canceled'])
    
    
    # Plot les résultats sous forme de courbe
    plt.figure(figsize=(15, 8))
    plt.plot(cancelled_percentages.index, cancelled_percentages, marker='o', color='red', linestyle='-', label="Taux d'annulation")
    
    plt.title("Pourcentage de réservations annulées par date d'arrivée")
    plt.xlabel("Proportion relative de réservations annulées par le passé")
    plt.ylabel("Pourcentage de réservations annulées")
    plt.xticks(rotation=45, ha='right')
    plt.savefig('databaseAnalysisGraphs/realtiveCancelByDates.png')
    plt.show()

def canceledByPreviousRes():
    filtered_df = df[df['repeated_guest'] == 1] #filtrage
    grouped_data = filtered_df.groupby('no_of_previous_cancellations')['booking_status'].value_counts(normalize=True).unstack().fillna(0)
    cancelled_percentages = grouped_data['Canceled'] * 100/(grouped_data['Canceled']+grouped_data['Not_Canceled'])
    
    
    # Plot les résultats sous forme de courbe
    plt.figure(figsize=(15, 8))
    plt.bar(cancelled_percentages.index.astype(str), cancelled_percentages, color='red', label="Taux d'annulation")
    
    plt.title("Taux d'annulation en fonction du nombre de réservations passées annulées")
    plt.xlabel("Nombre de réservations passées annulées")
    plt.ylabel("Pourcentage de réservations annulées")
    plt.xticks(rotation=45, ha='right')
    plt.savefig('databaseAnalysisGraphs/cancelByNbPreviousCanceled.png')
    plt.show()


def cancelationByMenuType():
    #calcule le nombre de cancellation par type de menu différent
    grouped_data = df.groupby('type_of_meal_plan')['booking_status'].value_counts(normalize=True).unstack().fillna(0)
    cancelled_percentages = grouped_data['Canceled']*100

    # Plot les résultats sous forme de courbe
    plt.figure(figsize=(15, 8))
    cancelled_percentages.plot(kind='bar', color='red')    
    plt.title("Pourcentage de réservations annulées par type de menu choisi")
    plt.xlabel("type de menu")
    plt.ylabel("nombre d'annulation correspondant")
    plt.xticks(rotation=45, ha='right')
    plt.savefig('databaseAnalysisGraphs/cancelByMenu.png')
    plt.show()

def menuChoiceRepartition():
    meal_plan_counts = df['type_of_meal_plan'].value_counts()

    # Plotting the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(meal_plan_counts, labels=meal_plan_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightgreen', 'lightcoral'])
    plt.title('Répartition des choix des menus')
    plt.savefig('databaseAnalysisGraphs/mealPlanDistributionPieChart.png')
    plt.show()

def cancelationByParkingLot():
    #calcule le nombre de cancellation par type de menu différent
    grouped_data = df.groupby('required_car_parking_space')['booking_status'].value_counts(normalize=True).unstack().fillna(0)
    cancelled_percentages = grouped_data['Canceled']*100/(grouped_data['Canceled']+grouped_data['Not_Canceled'])

    # Plot les résultats sous forme de courbe
    plt.figure(figsize=(15, 8))
    cancelled_percentages.plot(kind='bar', color='red')    
    plt.title("Pourcentage d'annulation par nombre de place de parking demandées")
    plt.xlabel("Nombre de place de parking demandées")
    plt.ylabel("Pourcentage d'annulation correspondant")
    plt.xticks(rotation=45, ha='right')
    plt.savefig('databaseAnalysisGraphs/cancelByParkingLot.png')
    plt.show()

def parkingLotRepartition():
    meal_plan_counts = df['required_car_parking_space'].value_counts()

    # Plotting the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(meal_plan_counts, labels=meal_plan_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightgreen', 'lightcoral'])
    plt.title('Répartition de la demande pour une place de parking')
    plt.savefig('databaseAnalysisGraphs/parkingDistribution.png')
    plt.show()
def cancelationByRoomType():
    #calcule le nombre de cancellation par type de menu différent
    grouped_data = df.groupby('room_type_reserved')['booking_status'].value_counts(normalize=True).unstack().fillna(0)
    cancelled_percentages = grouped_data['Canceled']*100

    # Plot les résultats sous forme de courbe
    plt.figure(figsize=(15, 8))
    cancelled_percentages.plot(kind='bar', color='red')    
    plt.title("Pourcentage d'annulation par type de chambre réservée")
    plt.xlabel("Type de chambre")
    plt.ylabel("Pourcentage d'annulation correspondant")
    plt.xticks(rotation=45, ha='right')
    plt.savefig('databaseAnalysisGraphs/cancelByRoomType.png')
    plt.show()

def menuChoiceRepartition():
    room_type_counts = df['room_type_reserved'].value_counts()

    # Plotting the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(room_type_counts, labels=room_type_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightgreen', 'lightcoral'])
    plt.title('Répartition du choix des types de chambres')
    plt.savefig('databaseAnalysisGraphs/roomTypeDistributionPieChart.png')
    plt.show()

def cancelationNbSpeReq():
    grouped_data = df.groupby('no_of_special_requests')['booking_status'].value_counts(normalize=True).unstack().fillna(0)
    cancelled_percentages = grouped_data['Canceled'] * 100/(grouped_data['Canceled']+grouped_data['Not_Canceled'])
    
    
    # Plot les résultats sous forme de courbe
    plt.figure(figsize=(15, 8))
    plt.plot(cancelled_percentages.index, cancelled_percentages, marker='o', color='red', linestyle='-', label="Taux d'annulation")
    
    plt.title("Pourcentage de réservations annulées nombre de requêtes spéciales")
    plt.xlabel("Nombre de requêtes spéciales")
    plt.ylabel("Pourcentage de réservations annulées")
    plt.xticks(rotation=45, ha='right')
    plt.savefig('databaseAnalysisGraphs/cancelByNbSpeReq.png')
    plt.show()

def nightRepartition():
    # Group by the number of adults and number of children
    grouped_data = df.groupby(['no_of_week_nights', 'no_of_weekend_nights'])['booking_status'].value_counts(normalize=True).unstack().fillna(0)
    
    # Select only the 'Cancelled' column
    cancelled_amount = grouped_data['Canceled']*100/(grouped_data['Canceled']+grouped_data['Not_Canceled']) #percentage of cancellation for each group
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    cancelled_amount.plot(kind='bar', color='red')
    
    plt.title('Pourcentage de réservation annulée en fonction de la répartition nuit semaine/week-end')
    plt.xlabel("Nombre de nuit en semaine, Nombre de nuit en week-end")
    plt.ylabel("Pourcentage de réservation annulées")
    plt.xticks(rotation=45, ha='right')
    plt.savefig('databaseAnalysisGraphs/cancelNightRep.png')
    plt.show()

def nightWeekRepartition():
    grouped_data = df.groupby('no_of_week_nights')['booking_status'].value_counts(normalize=True).unstack().fillna(0)
    cancelled_amount = grouped_data['Canceled']*100/(grouped_data['Canceled']+grouped_data['Not_Canceled']) #percentage of cancellation for each group
    
    # Calcul de la régression linéaire
    x = cancelled_amount.index.values
    y = cancelled_amount.values
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    
    # erreur standard
    residuals = y - y_pred
    ssr = np.sum(residuals ** 2)
    slope_error = np.sqrt(ssr / (len(x) - 2) / np.sum((x - np.mean(x)) ** 2)) #erreur standard de la pente

    # Tracer le graphique avec la régression linéaire
    plt.figure(figsize=(12, 8))
    cancelled_amount.plot(kind='bar', color='red')
    plt.plot(x, slope * x + intercept, color='blue', linestyle='--', linewidth=2, label=f'Regression linéaire (coefficient directeur={slope:.2f} ± {slope_error:.2f})')

    plt.title('Pourcentage de réservation annulée en fonction du nombre de nuit en semaine')
    plt.xlabel("Nombre de nuit en semaine")
    plt.ylabel("Pourcentage de réservation annulées")
    plt.xticks(rotation=45, ha='right')
    plt.legend() 
    plt.savefig('databaseAnalysisGraphs/cancelNightRepWeek.png')
    plt.show()

def nightWeekERepartition():
    grouped_data = df.groupby('no_of_weekend_nights')['booking_status'].value_counts(normalize=True).unstack().fillna(0)
    cancelled_amount = grouped_data['Canceled']*100/(grouped_data['Canceled']+grouped_data['Not_Canceled'])
    
    # Calcul de la régression linéaire
    x = cancelled_amount.index.values
    y = cancelled_amount.values
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    
    # erreur standard
    residuals = y - y_pred
    ssr = np.sum(residuals ** 2)
    slope_error = np.sqrt(ssr / (len(x) - 2) / np.sum((x - np.mean(x)) ** 2)) #erreur standard de la pente

    # Tracer le graphique avec la régression linéaire
    plt.figure(figsize=(12, 8))
    cancelled_amount.plot(kind='bar', color='red')
    plt.plot(x, slope * x + intercept, color='blue', linestyle='--', linewidth=2, label=f'Regression linéaire (coefficient directeur={slope:.2f} ± {slope_error:.2f})')

    plt.title('Pourcentage de réservation annulée en fonction du nombre de nuit en weekend')
    plt.xlabel("Nombre de nuit en weekend")
    plt.ylabel("Pourcentage de réservation annulées")
    plt.xticks(rotation=45, ha='right')
    plt.legend() 
    plt.savefig('databaseAnalysisGraphs/cancelNightRepWE.png')
    plt.show()

def cancelationPerMarketSeg():
    grouped_data = df.groupby('market_segment_type')['booking_status'].value_counts(normalize=True).unstack().fillna(0)
    cancelled_amount = grouped_data['Canceled'] * 100/(grouped_data['Canceled']+grouped_data['Not_Canceled'])
    
    # Plot les résultats sous forme de courbe
    plt.figure(figsize=(15, 8))
    cancelled_amount.plot(kind='bar', color='red')
    plt.title("Pourcentage de réservations annulées par segment de marché")
    plt.xlabel("Type de marché")
    plt.ylabel("Pourcentage de réservations annulées")
    plt.xticks(rotation=45, ha='right')
    plt.savefig('databaseAnalysisGraphs/cancelByMarketSeg.png')
    plt.show()

def cancelationRoomPrice():
    bins = pd.cut(df['avg_price_per_room'], bins=range(0, int(df['avg_price_per_room'].max()) + 11, 10)) #regroupe les valeurs de prix

    # Group the data by the bins
    grouped_data = df.groupby(bins)['booking_status'].value_counts(normalize=True).unstack().fillna(0)

    # Calculate the cancellation percentages
    cancelled_percentages = grouped_data['Canceled'] * 100 / (grouped_data['Canceled'] + grouped_data['Not_Canceled'])

    # Plot the results as a bar chart
    plt.figure(figsize=(15, 8))
    plt.bar(cancelled_percentages.index.astype(str), cancelled_percentages, color='red', label="Taux d'annulation")

    plt.title("Pourcentage de réservations annulées en fonction du prix moyen")
    plt.xlabel("Prix moyen pour chaque chambre réservée")
    plt.ylabel("Pourcentage de réservations annulées")
    plt.xticks(rotation=45, ha='right')
    plt.savefig('databaseAnalysisGraphs/cancelByAvgPrice.png')
    plt.show()

def marketSegRepartition():
    meal_plan_counts = df['market_segment_type'].value_counts()

    # Plotting the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(meal_plan_counts, labels=meal_plan_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightgreen', 'lightcoral'])
    plt.title('Répartition de la demande par type de marché')
    plt.savefig('databaseAnalysisGraphs/marketSegDistribution.png')
    plt.show()

def canceledRepartition():
    meal_plan_counts = df['booking_status'].value_counts()

    # Plotting the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(meal_plan_counts, labels=meal_plan_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightgreen', 'lightcoral'])
    plt.title('Pourcentage annulé global')
    plt.savefig('databaseAnalysisGraphs/globalCanceledDstribution.png')
    plt.show()