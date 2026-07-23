import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from geopy.geocoders import Nominatim
from geopy.distance import geodesic



raw_df = pd.read_csv("Flights1_2019_1.csv")

raw_df['ORIGIN_STATE_ABR'] = raw_df['ORIGIN_CITY_NAME'].str[-2:]

raw_df['DAY_OF_WEEK_STR'] = np.where(raw_df['DAY_OF_WEEK'] == 1, 'Mon', raw_df['DAY_OF_WEEK'])
raw_df['DAY_OF_WEEK_STR'] = np.where(raw_df['DAY_OF_WEEK'] == 2, 'Tue', raw_df['DAY_OF_WEEK_STR'])
raw_df['DAY_OF_WEEK_STR'] = np.where(raw_df['DAY_OF_WEEK'] == 3, 'Wed', raw_df['DAY_OF_WEEK_STR'])
raw_df['DAY_OF_WEEK_STR'] = np.where(raw_df['DAY_OF_WEEK'] == 4, 'Thu', raw_df['DAY_OF_WEEK_STR'])
raw_df['DAY_OF_WEEK_STR'] = np.where(raw_df['DAY_OF_WEEK'] == 5, 'Fri', raw_df['DAY_OF_WEEK_STR'])
raw_df['DAY_OF_WEEK_STR'] = np.where(raw_df['DAY_OF_WEEK'] == 6, 'Sat', raw_df['DAY_OF_WEEK_STR'])
raw_df['DAY_OF_WEEK_STR'] = np.where(raw_df['DAY_OF_WEEK'] == 7, 'Sun', raw_df['DAY_OF_WEEK_STR'])

raw_df['DEP_DELAY_NEW'] = np.where(raw_df['DEP_DELAY'] <= 0, 0, raw_df['DEP_DELAY'])
raw_df['DEP_DEL15'] = np.where(raw_df['DEP_DELAY'] >= 15, 1, 0)
raw_df['DEP_DEL15'] = np.where(raw_df['DEP_DELAY'] .isna(), np.nan, raw_df['DEP_DELAY'])

print(raw_df.head())


df = raw_df

df = df.dropna(subset = "DEP_DELAY")
df = df.dropna(subset = "ARR_TIME")
df = df.dropna(subset = "ARR_DELAY")


print(df.isna().sum())



df["flight_path"] = df["ORIGIN_CITY_NAME"] + " -> " + df["DEST_CITY_NAME"]

flight_path_df = pd.DataFrame(
    {
        "number_of_flights": df.groupby('flight_path')["ARR_DELAY"].count(),
        "mean_departure_delay": df.groupby('flight_path')["DEP_DELAY"].mean(),
        "mean_arrival_delay": df.groupby('flight_path')["ARR_DELAY"].mean()
    }
)

airport_df = pd.DataFrame(
    {

        "number_of_flights": df.groupby('ORIGIN_CITY_NAME')["ARR_DELAY"].count(),
        "mean_departure_delay": df.groupby('ORIGIN_CITY_NAME')["DEP_DELAY"].mean(),
        "mean_arrival_delay": df.groupby('ORIGIN_CITY_NAME')["ARR_DELAY"].mean()
    }


)

print(flight_path_df)
print(airport_df)

