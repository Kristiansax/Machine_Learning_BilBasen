import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC


# Importing data
basen = pd.read_csv('bilbasen3.csv', sep=';', keep_default_na=False, na_values=['-', 'Ring'])

# Setting new column names, Horsepower = horsepower, km = kilometers run, kml = kilometers per liter,
# acceleration = acceleration time in seconds, price = kr, year = year made
basen.columns = ['manufacturer', 'horsepower', 'km', 'kml', 'acceleration', 'price', 'year']

# Replacing NaN
basen = basen.ffill()

# Removing 'kr.' and '.' from price
basen['price'] = basen['price'].str.strip()
basen.price = [str(price).replace(str(price), str(price)[:-4]) if price is not 'nan' else '' for price in basen.price]
basen['price'] = basen['price'].str.replace('.', '')
basen.price = basen.price.astype(np.int64)

# Removing HK from horsepower
basen['horsepower'] = basen['horsepower'].str.strip()
basen.horsepower = [str(horsepower).replace(str(horsepower), str(horsepower)[:-3]) if horsepower is not 'nan' else ''
                    for horsepower in basen.horsepower]
basen.horsepower = basen.horsepower.astype(np.int64)

# Removing km/l from kml
basen.kml = basen.kml.str.strip()
basen.kml = [str(kml).replace(str(kml), str(kml)[:-5]) if kml is not 'nan' else '' for kml in basen.kml]
basen.kml = basen.kml.str.replace(',', '.')
basen.kml = basen.kml.astype(np.float64)

# Removing sek from acceleration
basen.acceleration = basen.acceleration.str.strip()
basen.acceleration = [str(acceleration).replace(str(acceleration), str(acceleration)[:-5]) if acceleration is not 'nan'
                      else '' for acceleration in basen.acceleration]
basen.acceleration = basen.acceleration.str.replace(',', '.')
basen.acceleration = basen.acceleration.astype(np.float64)

# Scatter Matrix
pd.plotting.scatter_matrix(basen, c=basen['price'], figsize=[6, 6], s=150, marker='D')
plt.show()

# Splitting name to categories
manuf = []

for s in basen.manufacturer:
    s = s.replace('.', ' ')
    ss = s.split(' ')

    manuf.append(ss[0])

for i, m in enumerate(manuf):
    if 'Citroxebn' in m:
        manuf[i] = m.replace('Citroxebn', 'Citroen')

# Replacing the manufacturer column and making it categorical
basen.manufacturer = manuf
basen_categorical = pd.get_dummies(basen, drop_first=True)

# Setup pipelines and test/train sets
# KNN
steps_knn = [('scaling', StandardScaler()),
             ('KNN', KNeighborsRegressor())]
knn_pipeline = Pipeline(steps_knn)

# LinearRegression
steps_lr = [('scaling', StandardScaler()),
            ('linear regression', LinearRegression())]
linearregression_pipeline = Pipeline(steps_lr)

# SVC
steps_svc = [('scaling', StandardScaler()),
             ('SVC', SVC())]
scv_pipeline = Pipeline(steps_svc)

X = basen_categorical.drop('price', axis=1)
y = basen_categorical['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Scaled
knn_scaled = knn_pipeline.fit(X_train, y_train)
linearregression_scaled = linearregression_pipeline.fit(X_train, y_train)
svc_scaled = scv_pipeline.fit(X_train, y_train)

# Unscaled
knn_unscaled = KNeighborsRegressor().fit(X_train, y_train)
linearregression_unscaled = LinearRegression().fit(X_train, y_train)
svc_unscaled = SVC().fit(X_train, y_train)

# Compare with metrics
print('----------- Accuracy with .score -----------')
print('KNN Scaled accuracy: {}'.format(knn_scaled.score(X_test, y_test)))
print('KNN Unscaled accuracy: {}'.format(knn_unscaled.score(X_test, y_test)))
print()
print('Linear Regression Scaled accuracy: {}'.format(linearregression_scaled.score(X_test, y_test)))
print('Linear Regression Unscaled accuracy: {}'.format(linearregression_unscaled.score(X_test, y_test)))
print()
print('SVC Scaled accuracy: {}'.format(svc_scaled.score(X_test, y_test)))
print('SVC Unscaled accuracy: {}'.format(svc_unscaled.score(X_test, y_test)))
print()

# Cross Validation score
print('------- Accuracy with cross_val_score -------')
print('KNN Scaled accuracy: {}'.format(cross_val_score(knn_scaled, X_test, y_test, cv=5, scoring='r2').mean()))
print('KNN Unscaled accuracy: {}'.format(cross_val_score(knn_unscaled, X_test, y_test, cv=5, scoring='r2').mean()))
print()
print('Linear Regression Scaled accuracy: {}'
      .format(cross_val_score(linearregression_scaled, X_test, y_test, cv=5, scoring='r2').mean()))
print('Linear Regression Unscaled accuracy: {}'
      .format(cross_val_score(linearregression_unscaled, X_test, y_test, cv=5, scoring='r2').mean()))
print()
print('SVC Scaled accuracy: {}'.format(cross_val_score(svc_scaled, X_test, y_test, cv=3, scoring='r2').mean()))
print('SVC Unscaled accuracy: {}'.format(cross_val_score(svc_unscaled, X_test, y_test, cv=3, scoring='r2').mean()))
