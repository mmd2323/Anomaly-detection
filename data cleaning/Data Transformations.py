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



df = pd.read_csv("Flights1_2019_1.csv")


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





