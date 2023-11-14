# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def plotDecisionTree(model, X, y):
    plt.figure(figsize=(100, 100))
    plot_tree(model, filled=True, feature_names=X.columns.tolist(), 
              class_names=y.unique().astype(str).tolist(), rounded=True)
    plt.show()

def plotModels(trained_models, X, y):
    for model in trained_models:
        if model == "Decision Tree":
            plotDecisionTree(trained_models[model], X, y)