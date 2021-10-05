import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
import lightgbm as lgb

from sklearn.metrics import mean_squared_error

data = pd.read_csv("hyundai.csv")

unneeded_columns = ['tax(£)', 'engineSize']
data = data.drop(unneeded_columns, axis=1)

def onehot_encode(df, colums, prefixes):
    df = df.copy()
    for column, prefix in zip(colums, prefixes):
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df

data = onehot_encode(
    data, ["model", "year", "transmission", "mileage", "fuelType", "mpg"],
    ["model", "year", "trans", "mile", "fuel", "mpg"]
)

y = data.loc[:, 'price']
x = data.drop('price', axis=1)

scalar = StandardScaler()

x = scalar.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=34)

lgb_model = lgb.LGBMRegressor(
    boosting_type='gbdt',
    num_leaves=31,
    n_estimators=100,
    reg_lambda=1.0
)

lgb_model.fit(x_train, y_train)

lgb_y_preds = lgb_model.predict(x_test)

def predict_price(model, year, transmission, mileage, fueltype, mpg):
    temp_list = [[model, year, transmission, mileage, fueltype, mpg]]
    data_test = pd.DataFrame(temp_list, columns =["model", "year", "transmission", "mileage", "fuelType", "mpg"])
    data_test = onehot_encode(data_test, ["model", "year", "transmission", "mileage", "fuelType", "mpg"],
    ["model", "year", "trans", "mile", "fuel", "mpg"])
    return lgb_model.predict(data_test)

predict_price("I20", 2017, "Automatic", 15000, "Petrol", 55)
