import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score
# Sample dataset (replace with your dataset)
#custom_column_names = ["ID", "adultsCount", "ChildrenCount", "WENightsCount", "WeekNightsCount", "MealType", "ParkingSpaceRequired", "RoomType", "LeadTime", "ArrivalYear", "ArrivalMonth", "ArrivalDay", "MarketSegmentType", "RepeatedGuest", "NpPreviousCancellations"]
df = pd.read_csv('data/Hotel_Reservations.csv')

# Define features and target
features = df.columns.difference(['Booking_ID', 'booking_status'])


categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', "booking_status"]

encoder = OrdinalEncoder()
df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

X = df[features]
y = df['booking_status']

# Split the data into train and test sets
"""
# Define classifiers
classifiers = {
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Define hyperparameter grids for grid search
param_grids = {
    'KNN': {
        'n_neighbors': [3, 5, 7],
        'p': [1, 2]
    },
    'Decision Tree': {
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    },
    'Random Forest': {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 5, 10]
    }
}

# Define the SVM pipeline and parameter grid
svm_pipeline = Pipeline(steps=[
    #('preprocessor', preprocessor),
    ('classifier', SVC())
])

svm_param_grid = {
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__kernel': ['linear', 'rbf', 'poly'],
}

# Define the hierarchical clustering pipeline and parameter grid
hclust_pipeline = Pipeline(steps=[
    #('preprocessor', preprocessor),
    ('classifier', AgglomerativeClustering())
])

hclust_param_grid = {
    'classifier__n_clusters': [2, 3, 4],
    'classifier__affinity': ['euclidean', 'manhattan'],
    'classifier__linkage': ['ward', 'complete'],
}

results = {}

# Perform grid search for each classifier and hyperparameters
for name, classifier in classifiers.items():
    clf = GridSearchCV(classifier, param_grids[name], scoring='accuracy', cv=3)
    clf.fit(X_train, y_train)
    best_estimator = clf.best_estimator_
    
    # Evaluate the best estimator on the test set
    y_pred = best_estimator.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'best_estimator': best_estimator,
        'accuracy': accuracy
    }

# Print the results
for name, result in results.items():
    print(f'{name} Accuracy: {result["accuracy"]:.2f}')
    print(f'Best Estimator for {name}:\n{result["best_estimator"]}\n')
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a list of classifiers
classifiers = [
    #('KNN', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('HCLUST', AgglomerativeClustering()),  # HCLUST for clustering
    ('SVM', SVC())
]

# Define hyperparameter grids for grid search for each classifier
param_grids = {
    #'KNN': {
    #    'classifier__n_neighbors': [3, 5, 7],
    #    'classifier__p': [1, 2]
    #},
    'Decision Tree': {
        'classifier__max_depth': [None, 5, 10],
        'classifier__min_samples_split': [2, 5]
    },
    'Random Forest': {
        'classifier__n_estimators': [10, 50, 100],
        'classifier__max_depth': [None, 5, 10]
    },
    'HCLUST': {},  # No hyperparameters for HCLUST (clustering)
    'SVM': {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__kernel': ['linear', 'rbf', 'poly'],
    }
}

results1 = {}
results2 = {}

# Create a custom pipeline for each classifier and perform grid search over 10 folds
for name, classifier in classifiers:
    clf_pipeline = Pipeline(steps=[
        #('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    # Perform grid search with 10-fold cross-validation
    grid_search = GridSearchCV(clf_pipeline, param_grids[name], scoring='accuracy', cv=10)
    scores = cross_val_score(grid_search, X_train, y_train, cv=10)
    results1[name] = scores

    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_
    
    # Evaluate the best estimator on the test set
    y_pred = best_estimator.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results2[name] = {
        'best_estimator': best_estimator,
        'accuracy': accuracy
    }
    

# Print the results
for name, scores in results1.items():
    print(f'{name} Mean Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})')

for name, result in results2.items():
    print(f'{name} Accuracy: {result["accuracy"]:.2f}')
    print(f'Best Estimator for {name}:\n{result["best_estimator"]}\n')
