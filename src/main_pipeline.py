import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dataset (replace with your dataset)
#custom_column_names = ["ID", "adultsCount", "ChildrenCount", "WENightsCount", "WeekNightsCount", "MealType", "ParkingSpaceRequired", "RoomType", "LeadTime", "ArrivalYear", "ArrivalMonth", "ArrivalDay", "MarketSegmentType", "RepeatedGuest", "NpPreviousCancellations"]
df = pd.read_csv('data/Hotel_Reservations.csv')
print(df.head(1))

# Define features and target
features = df.columns.difference(['Booking_ID', 'booking_status'])


categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', "booking_status"]

encoder = OrdinalEncoder()
df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
print(df)

X = df[features]
y = df['booking_status']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
print(X_train.flags)
# Define the KNN classifier
knn_classifier = KNeighborsClassifier()

# Define hyperparameter grid for grid search
param_grid = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
}

# Perform grid search for the KNN classifier
grid_search = GridSearchCV(knn_classifier, param_grid, scoring='accuracy', cv=3)
grid_search.fit(X_train, y_train)

# Get the best estimator
best_knn = grid_search.best_estimator_

# Evaluate the best KNN estimator on the test set
y_pred = best_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print the best KNN estimator and accuracy
print('Best KNN Estimator:', best_knn)
print(f'Accuracy: {accuracy:.2f}')
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
