import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dataset (replace with your dataset)
data = "etset"

df = pd.DataFrame(data)

# Define features and target
features = df.columns.difference(['Booking_ID', 'booking_status'])
X = df[features]
y = df['booking_status']

# Define column transformer
numeric_features = X.columns.difference(['room_type_reserved', 'market_segment_type'])
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_features = ['room_type_reserved', 'market_segment_type']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers = {
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Define hyperparameter grids for grid search
param_grids = {
    'KNN': {
        'classifier__n_neighbors': [3, 5, 7],
        'classifier__p': [1, 2]
    },
    'Decision Tree': {
        'classifier__max_depth': [None, 5, 10],
        'classifier__min_samples_split': [2, 5]
    },
    'Random Forest': {
        'classifier__n_estimators': [10, 50, 100],
        'classifier__max_depth': [None, 5, 10]
    }
}

results = {}

# Create a custom pipeline for each classifier and perform grid search
for name, classifier in classifiers.items():
    clf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    # Perform grid search
    grid_search = GridSearchCV(clf_pipeline, param_grids[name], scoring='accuracy', cv=3)
    grid_search.fit(X_train, y_train)
    
    # Get the best estimator
    best_estimator = grid_search.best_estimator_
    
    # Evaluate the best estimator on the test set
    y_pred = best_estimator.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

# Print the results
for name, accuracy in results.items():
    print(f'{name} Accuracy: {accuracy:.2f}')
