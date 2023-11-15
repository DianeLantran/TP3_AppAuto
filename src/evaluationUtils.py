# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 23:42:39 2023

@author: solene
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

def mae(y_true, y_pred):
    return ((y_pred - y_true) ** 2).mean()


def rmse(y_true, y_pred):
    return np.sqrt(mae(y_true, y_pred))


def r2(y_true, y_pred):
    sse_m1 = ((y_pred-y_true) ** 2).sum()
    sse_mb = ((y_true.mean() - y_true) ** 2).sum()
    return 1 - sse_m1 / sse_mb


#x et y sont les caractéristiques et la variable cible
def plot_learning_curve(axis, model, X, y):
    train_errors = []
    validation_errors = []
    
    # Divise les données en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Essaye différentes tailles de l'ensemble d'entraînement
    for m in range(1, len(X_train) + 1):
        model.fit(X_train[:m], y_train[:m])  # Entraîne le modèle sur un sous-ensemble d'entraînement
    
        y_train_pred = model.predict(X_train[:m])
        train_error = metrics.mean_squared_error(y_train[:m], y_train_pred)
        train_errors.append(train_error)
    
        y_val_pred = model.predict(X_val)
        validation_error = metrics.mean_squared_error(y_val, y_val_pred)
        validation_errors.append(validation_error)
    
    axis.plot(range(1, len(X_train) + 1), train_errors, label="Training error")
    axis.plot(range(1, len(X_train) + 1), validation_errors, 
              linestyle='dashed', label="Erreur de validation")
    axis.set_xlabel("Taille du data set d entrainement")
    axis.set_ylabel("MSE")
    axis.legend()
    axis.set_title("Courbe apprentissage")
    axis.grid(True)
    return axis
    

def error_plot(axis, y_true, y_pred, model):
    axis.scatter(y_true, y_pred, c='b', label='Valeurs prédites')

    # Plot y = x    
    axis.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], '-', label='y = x')
    
    axis.set_xlabel("Valeurs attendues")
    axis.set_ylabel("Valeurs prédites")
    axis.set_title("Comparaison valeurs attendues vs prédites")
    axis.legend()
    axis.grid(True)
    return axis
    
    
def residual_plot(axis, y_true, y_predicted): 
    axis.plot(y_predicted, y_true -y_predicted, "*") 
    axis.plot(y_predicted, np.zeros_like(y_predicted), "-") 
    axis.legend(["Données", "Perfection"]) 
    axis.set_title("Plot résiduel") 
    axis.set_xlabel("Valeurs prédites") 
    axis.set_ylabel("Résiduelles") 
    return axis
    


def cross_validation_matrix(model, X, y, k):
    fold_size = len(X) // k
    cross_val_scores = []
    
    for i in range(k):
        # Divise les données en ensembles d'entraînement et de validation
        validation_start = i * fold_size
        validation_end = (i + 1) * fold_size
        
        X_test = X[validation_start:validation_end]
        y_test = y[validation_start:validation_end]
        
        X_train = np.concatenate([X[:validation_start], X[validation_end:]], axis=0)
        y_train = np.concatenate([y[:validation_start], y[validation_end:]], axis=0)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        score = r2(y_pred, y_test)
        cross_val_scores.append(score)
    df = pd.DataFrame({"Plis": range(1, len(cross_val_scores) + 1), "Score de Validation": cross_val_scores})
    # Affiche le DataFrame
    print(df)
    return cross_val_scores

def confusionMatrix(y_test, y_pred, name):
    # Calculer la matrice de confusion
    cm = metrics.confusion_matrix(y_test, y_pred)

    # Afficher la matrice de confusion avec seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Classe 0', 'Classe 1'],
                yticklabels=['Classe 0', 'Classe 1'])
    plt.xlabel('Predictions')
    plt.ylabel('True labels')
    plt.title(f'Confusion matrix for {name}')
    plt.show()

def scores(y_test, y_pred, name):
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    accuracy = metrics.accuracy_score(y_test, y_pred)

    print(f'Methode: {name}')
    print("Précision:", precision)
    print("Rappel:", recall)
    print("F1-score:", f1)
    
    report = metrics.classification_report(y_test, y_pred)
    print("Rapport de classification:")
    print(report)

def graphScores(resultsList):
    model_names = [result[0] for result in resultsList]
    precision_scores = [metrics.precision_score(result[1], result[2]) for result in resultsList]
    recall_scores = [metrics.recall_score(result[1], result[2]) for result in resultsList]
    f1_scores = [metrics.f1_score(result[1], result[2]) for result in resultsList]
    accuracy_scores = [metrics.accuracy_score(result[1], result[2]) for result in resultsList]

    # Largeur des barres
    bar_width = 0.15

    # Positions des barres pour chaque modèle
    index = np.arange(len(model_names))

    colors = ['#ff7473','#ffc952', '#47b8e0', '#34314c']

    # Tracer le graphique en barres groupées
    plt.figure(figsize=(12, 6))

    plt.bar(index, precision_scores, width=bar_width, label='Precision', color=colors[0])
    plt.bar(index + bar_width, recall_scores, width=bar_width, label='Recall', color=colors[1])
    plt.bar(index + 2 * bar_width, f1_scores, width=bar_width, label='F1-score', color=colors[2])
    plt.bar(index + 3 * bar_width, accuracy_scores, width=bar_width, label='Accuracy', color=colors[3])

    # Ajouter des détails au graphique
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('Scores', fontweight='bold')
    plt.title('Performances comparison for all models')
    plt.xticks(index + 1.5 * bar_width, model_names)
    plt.legend()

    plt.show()

def ROCAndAUC(resultsList):
    plt.figure(figsize=(8, 8))

    for result in resultsList:
        model_name = result[0]
        y_test = result[1]
        y_pred = result[2]

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        auc = metrics.roc_auc_score(y_test, y_pred)

        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
