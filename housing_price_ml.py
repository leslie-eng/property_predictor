import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge  # noqa F401
from sklearn.pipeline import make_pipeline
import pickle

def wrangle(filepath):
    # Read CSV file
    df = pd.read_csv(filepath)

    # Subset data: Apartments in "Capital Federal", less than 400,000
    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 400_000
    df = df[mask_ba & mask_apt & mask_price]

    # Subset data: Remove outliers for "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]

    # Split "lat-lon" column
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)

    # Get place name
    df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]
    df.drop(columns="place_with_parent_names", inplace=True)
    
    #dropping the columns with less than half of the series length
    df.drop(columns=["floor","expenses"], inplace=True)
    
    #dropping the low and high cardinality
    df.drop(columns=["operation","property_type","currency","properati_url"], inplace=True)
    
    #leakage colmns dropped
    df.drop(columns=[
        "price",
        "price_aprox_local_currency",
        "price_usd_per_m2",
        "price_per_m2"], inplace=True)
    # dropping multicollinearity features
    df.drop(columns=[
        "surface_total_in_m2",
        "rooms"
    ], inplace=True)

    
    return df

df = wrangle("data/housing_data.csv")

target = "price_aprox_usd"
y_train = df[target]
features = ["surface_covered_in_m2", "lat", "lon", "neighborhood"]
X_train = df[features]

y_mean = y_train.mean()
y_pred = [y_mean]* len(y_train)

model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    SimpleImputer(),
    Ridge(),
)
model.fit(X_train, y_train)

pickle.dump(model, open("house_price.pkl", "wb"))

