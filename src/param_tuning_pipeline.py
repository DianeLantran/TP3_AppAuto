from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import preprocessing as prep

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

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a list of classifiers
classifiers = [
    ('KNN', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('SVM', SVC()),
    ('Logistic Regression', LogisticRegression()),
    ('Naive Bayes', MultinomialNB()),
    ('Neural Network', MLPClassifier())  # Add MLPClassifier
]


# Define hyperparameter grids for grid search for each classifier
param_grids = {
    'KNN': {
        'classifier__n_neighbors': [2, 3, 4, 5, 7, 9, 15],
        'classifier__p': [1, 2]
    },
    
    'Decision Tree': {
        'classifier__max_depth': [None, 5, 10],
        'classifier__min_samples_split': [2, 5]
    },
    'Random Forest': {
        'classifier__n_estimators': [10, 50, 100, 250],
        'classifier__max_depth': [None, 5, 10]
    },
    'SVM': {
        'classifier__C': [0.1, 0.5, 1.0, 10.0, 15.0],
        'classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'classifier__degree': [2, 3, 5, 7],
    },
    'Logistic Regression': {
        'classifier__C': [0.1, 0.5, 1.0, 10.0],
    },
    'Naive Bayes': {
        'classifier__alpha': [1.0, 0.5, 0.1],
    },
    'Neural Network': {
        'classifier__hidden_layer_sizes': [(50, 50), (100, 50, 25)],
        'classifier__activation': ['relu', 'tanh'],
        'classifier__max_iter': [1000],
    }
}

# Create a custom pipeline for each classifier and perform grid search over 10 folds
for name, classifier in classifiers:
    if name in ('SVM', 'Naive Bayes', 'Logistic Regression'):
        clf_pipeline = Pipeline(steps=[
            ('classifier', classifier)
        ])
    else:
        clf_pipeline = Pipeline(steps=[
            ('pca', PCA(n_components=0.90)),
            ('classifier', classifier)
        ])

    # Perform grid search with 10-fold cross-validation
    grid_search = GridSearchCV(clf_pipeline, param_grids[name], scoring='accuracy', cv=10)
    scores = grid_search.fit(X, y)
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    print(f'Best Estimator for {name}:\n{best_params}')
    print(f'Best Accuracy for {name}: {best_accuracy:.2f}\n')
    
