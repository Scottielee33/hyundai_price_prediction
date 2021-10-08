import pandas as pd
import numpy as np
from sklearn import linear_model
import lightgbm as lgb

df1 = pd.read_csv("ford.csv")
df1["brand"] = "ford"
df2 = pd.read_csv("audi.csv")
df2["brand"] = "audi"
df3 = pd.read_csv("bmw.csv")
df3["brand"] = "bmw"
df4 = pd.read_csv("mercedes.csv")
df4["brand"] = "mercedes"
df5 = pd.read_csv("toyota.csv")
df5["brand"] = "toyata"
df6 = pd.read_csv("hyundai.csv")
df6["brand"] = "hyundai"
df = pd.concat([df1, df2, df3, df4, df5, df6])

def column_to_dictionary(column):
    lst = df[column].tolist()
    lst = list(set(lst))
    lst.sort()
    dictionary = dict(zip(lst, list(range(len(lst)+1))))
    return dictionary

model_dict = column_to_dictionary("model")
transmission_dict = column_to_dictionary("transmission")
fueltype_dict = column_to_dictionary("fuelType")
brand_dict = column_to_dictionary("brand")

inputs = df.drop(['tax(£)', 'price', 'tax', 'engineSize'], axis='columns')
target = df.price

inputs = inputs.replace(model_dict)
inputs = inputs.replace(transmission_dict)
inputs = inputs.replace(fueltype_dict)
inputs = inputs.replace(brand_dict)

lgb_model = lgb.LGBMRegressor(
    boosting_type='gbdt',
    num_leaves=31,
    n_estimators=100,
    reg_lambda=1.0
)

lgb_model.fit(inputs, target)
lgb_r2 = lgb_model.score(inputs.values, target)

pred = lgb_model.predict([[model, year, ]])