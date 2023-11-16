# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:26:08 2023

@author: basil
"""
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

def colToOrdinal(df, colnames):
    # Prepare l'encoder object
    encoder = OrdinalEncoder(encoded_missing_value=-1)

    # Fit et transforme la colonne selectionnée
    df[colnames] = encoder.fit_transform(df[colnames])
    return df


def standardize(df):
    # Standardizes data using mixmaxscaler
    scaler = MinMaxScaler()
    df_standardized = scaler.fit_transform(df)
    df_standardized = pd.DataFrame(df_standardized, columns=df.columns)
    return df_standardized

def preprocess(df, categorical_features, features, target):
    # Encodage des données :
    df = colToOrdinal(df, categorical_features)
    
    # Séparation en features/target
    X = df[features]
    y = df[target]

    # Standardise les données
    X = standardize(X)
    return X,y
    