from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

# Get dataset
df = pd.read_csv('../data/Hotel_Reservations.csv')
features = df.columns.difference(['Booking_ID', 'booking_status'])
categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', "booking_status"]

# Ordinal Encoding
encoder = OrdinalEncoder()
df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

# Standardization
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Set features & target sets
X = df[features]
y = df['booking_status']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a list of classifiers
classifiers = [
    ('Naive Bayes', MultinomialNB()),
    ('Neural Network', MLPClassifier())  # Add MLPClassifier
]

# Define hyperparameter grids for grid search for each classifier
param_grids = {
    'Naive Bayes': {
        'classifier__alpha': [1.0, 0.5, 0.1],
    },
    'Neural Network': {
        'classifier__hidden_layer_sizes': [(50, 50), (100, 50, 25)],
        'classifier__activation': ['relu', 'tanh'],
        'classifier__max_iter': [1000, 1500],
    }
}

# Create a custom pipeline for each classifier and perform grid search over 10 folds
for name, classifier in classifiers:
    if name in ('SVM', 'Naive Bayes'):
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
    
