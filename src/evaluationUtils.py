# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 23:42:39 2023

@author: solene
"""

import numpy as np
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

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

