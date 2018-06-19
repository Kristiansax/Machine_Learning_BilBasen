import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import Cleaning

# Importing Data
df = Cleaning.cleaned_df(categorical=True)

# Data and target values
X = df.drop('price', axis=1)
y = df['price']

# Test range of n_neighbors
k_range = range(1, 50)

# Array to hold scores
k_scores = []

# For loop evaluating estimator with each amount of n_neighbors
for k in k_range:
    steps_knn = [('imp', Imputer(missing_values=-1, strategy='mean', axis=0)),
                 ('scaling', StandardScaler()),
                 ('KNN', KNeighborsRegressor(n_neighbors=k))]
    knn_pipeline = Pipeline(steps_knn)

    scores = cross_val_score(knn_pipeline, X, y, cv=5, scoring='r2')

    k_scores.append(scores.mean())

# Plot showing the scores of each of the n_neighbors
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
