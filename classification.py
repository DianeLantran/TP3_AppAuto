# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:18:32 2023

@author: basil
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
classifiers = [
    ('KNN', KNeighborsClassifier(n_neighbors = 3, p = 1)),
    ('Decision Tree', DecisionTreeClassifier(max_depth = None, 
                                             min_samples_split = 2)),
    ('Random Forest', RandomForestClassifier(max_depth = None, 
                                             n_estimators = 100)),
    ('SVM', SVC(C = 15, degree = 2, kernel = "rbf")),
    ('Naive Bayes', MultinomialNB(alpha = 0.1)),
    ('Logistic Regression', LogisticRegression(C = 0.1, max_iter = 500)),
    ('Neural Network', MLPClassifier(activation = "relu", 
                                     hidden_layer_sizes = (100, 50, 25),
                                     max_iter = 1000))
    
]
def classify(X, y):
    trained_models = {}
    for name, classifier in classifiers:
        if name in ('SVM', 'Naive Bayes', "Decision Tree", "Random Forest",
                    "Logistic Regression"):
            pipeline = Pipeline(steps=[
                ('classifier', classifier)
            ])
        else:
            pipeline = Pipeline(steps=[
                ('pca', PCA(n_components=0.90)),
                ('classifier', classifier)
            ])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy for {name}: {accuracy:.2f}')
        trained_models[name] = pipeline.named_steps['classifier']
        pca = PCA(2)
        test_df = pca.fit_transform(X_test)
        colors_actual_correct = ['blue' if class_label == 0 else 'red' for class_label in y_pred[y_test == y_pred]]
        colors_actual_incorrect = ['blue' if class_label == 0 else 'red' for class_label in y_pred[y_test != y_pred]]
        plt.scatter(test_df[y_test == y_pred, 0], test_df[y_test == y_pred, 1], marker='o', c=colors_actual_correct, label='Correct Predictions')

        # Plot incorrectly predicted points with crosses
        plt.scatter(test_df[y_test != y_pred, 0], test_df[y_test != y_pred, 1], marker='x', c=colors_actual_incorrect, label='Incorrect Predictions')
        
        plt.title(f'Scatter Plot with PCA for {name}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()
    return trained_models