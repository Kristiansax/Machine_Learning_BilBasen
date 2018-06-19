import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer


def cleaned_df(categorical=False):
    # Importing data
    basen = pd.read_csv('bilbasen3.csv', sep=';', keep_default_na=False, na_values=['-'])

    # Setting new column names, Horsepower = horsepower, km = kilometers run, kml = kilometers per liter,
    # acceleration = acceleration time in seconds, price = kr, year = year made
    basen.columns = ['manufacturer', 'horsepower', 'km', 'kml', 'acceleration', 'price', 'year']

    # Dropping 'Ring' in Price
    basen = basen[basen.price != 'Ring']
    basen = basen.fillna(value=-1)

    # Removing 'kr.' and '.' from price
    basen.price = [str(price).replace(str(price), str(price)[:-4]) if price is not -1 else print('-1') for price in
                   basen.price]
    basen['price'] = basen['price'].str.replace('.', '')
    basen.price = basen.price.astype(np.int64)

    # Removing HK from horsepower
    basen.horsepower = [str(horsepower).replace(' HK', '') for horsepower in basen.horsepower]
    basen.horsepower = basen.horsepower.astype(np.int64)

    # Removing km/l from kml
    basen.kml = [str(kml).replace(str(kml), str(kml)[:-5]) if kml is not -1 else print('-1') for kml in basen.kml]
    basen.kml = basen.kml.str.replace(',', '.')
    basen.kml = basen.kml.astype(np.float64)

    # Removing sek from acceleration
    basen.acceleration = [
        str(acceleration).replace(str(acceleration), str(acceleration)[:-5]) if acceleration is not -1
        else print('-1') for acceleration in basen.acceleration]
    basen.acceleration = basen.acceleration.str.replace(',', '.')
    basen.acceleration = basen.acceleration.astype(np.float64)

    # Splitting name to categories
    manuf = []

    for s in basen.manufacturer:
        s = s.replace('.', ' ')
        ss = s.split(' ')

        manuf.append(ss[0])

    for i, m in enumerate(manuf):
        if 'Citroxebn' in m:
            manuf[i] = m.replace('Citroxebn', 'Citroen')
    basen.manufacturer = manuf
    basen_categorical = pd.get_dummies(basen, drop_first=True)

    if categorical:
        return basen_categorical
    else:
        return basen
