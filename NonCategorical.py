from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import Cleaning

df = Cleaning.cleaned_df(categorical=False)

X = df.drop(['price', 'manufacturer'], axis=1)
y = df['price']

# KNN Pipeline
steps_knn = [('imp', Imputer(missing_values=-1, strategy='mean', axis=0)),
             ('scaling', StandardScaler()),
             ('KNN', KNeighborsRegressor(n_neighbors=2))]
knn_pipeline = Pipeline(steps_knn)

# LinearRegression Pipeline
steps_lr = [('imp', Imputer(missing_values=-1, strategy='mean', axis=0)),
            ('linear regression', LinearRegression())]
lr_pipeline = Pipeline(steps_lr)

print('KNN accuracy: {}'.format(cross_val_score(knn_pipeline, X, y, cv=5, scoring='r2').mean()))
print()
print('Linear Regression accuracy: {}'.format(cross_val_score(lr_pipeline, X, y, cv=5, scoring='r2').mean()))
