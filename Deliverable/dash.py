"""
Flight Anomaly Control Tower
=====================================
Interactive Streamlit dashboard for detecting anomalous flights in the
RITA / BTS Reporting Carrier On-Time Performance dataset (January 2019 extract).

Actual columns in this dataset:
    YEAR, DAY_OF_WEEK, FL_DATE, ORIGIN_AIRPORT_ID, ORIGIN_AIRPORT_SEQ_ID,
    ORIGIN_CITY_MARKET_ID, ORIGIN_CITY_NAME, DEST_AIRPORT_ID, DEST_AIRPORT_SEQ_ID,
    DEST_CITY_MARKET_ID, DEST_CITY_NAME, DEST_STATE_ABR, DEP_DELAY, ARR_TIME,
    ARR_DELAY, ARR_DELAY_NEW, ARR_DEL15

Note: there is no carrier, distance, taxi time, or elapsed-time column in this
extract, so the app below is built entirely around delay, timing, and route fields.

Run with:
    streamlit run streamlit_app.py

Requirements (pip install):
    streamlit pandas numpy scikit-learn plotly scipy
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import statsmodels.api as sm

# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="Flight Anomaly Control Tower",
    layout="wide",
    page_icon="\u2708\ufe0f",
)

# ======================================================================
# DATA LOADING / CLEANING
# ======================================================================
@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file, low_memory=False)
    # Drop any trailing "Unnamed: N" junk columns from stray commas in the export
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


@st.cache_data(show_spinner=False)
def preprocess(df, sample_n, random_state=42):
    d = df.copy()

    d["FL_DATE"] = pd.to_datetime(d["FL_DATE"], errors="coerce")

    # Rows with no DEP_DELAY / ARR_DELAY are (almost certainly) cancelled or
    # diverted flights -- there's no explicit CANCELLED column in this extract,
    # so missingness itself is the signal. Flag them before dropping.
    d["LIKELY_CANCELLED"] = d["DEP_DELAY"].isna() | d["ARR_DELAY"].isna()

    # Keep a separate copy of cancelled-flight counts for the EDA tab
    cancelled_summary = d.groupby("DAY_OF_WEEK")["LIKELY_CANCELLED"].mean()

    # For all modeling / numeric analysis we need complete rows
    numeric_cols = ["DEP_DELAY", "ARR_DELAY", "ARR_DELAY_NEW", "ARR_DEL15", "ARR_TIME"]
    d = d.dropna(subset=numeric_cols)

    for c in numeric_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=numeric_cols)

    # Derived features
    d["ARR_HOUR"] = (d["ARR_TIME"] // 100).clip(0, 23)
    d["DELAY_GAP"] = d["ARR_DELAY"] - d["DEP_DELAY"]  # delay picked up / made up in flight
    d["ROUTE"] = d["ORIGIN_CITY_NAME"] + " \u2192 " + d["DEST_CITY_NAME"]

    if len(d) > sample_n:
        d = d.sample(sample_n, random_state=random_state)

    return d.reset_index(drop=True), cancelled_summary

def transform_flightpath(df):
    df["flight_path"] = df["ORIGIN_CITY_NAME"] + " -> " + df["DEST_CITY_NAME"]

    flight_path_df = pd.DataFrame(
        {
            "number_of_flights": df.groupby('flight_path')["ARR_DELAY"].count(),
            "mean_departure_delay": df.groupby('flight_path')["DEP_DELAY"].mean(),
            "probablility_of_dep_delay": df.groupby('flight_path').apply(lambda x: (x['DEP_DELAY'] < 0).mean()),
            "max_departure_delay": df.groupby('flight_path')["DEP_DELAY"].max(),
            "mean_arrival_delay": df.groupby('flight_path')["ARR_DELAY"].mean(),
            "probablility_of_arr_delay": df.groupby('flight_path').apply(lambda x: (x['ARR_DELAY'] < 0).mean()),
            "max_arrival_delay": df.groupby('flight_path')["ARR_DELAY"].max(),

        }
    )

    return flight_path_df

def transform_airport(df):
    airport_df = pd.DataFrame(
        {

            "number_of_flights": df.groupby('ORIGIN_CITY_NAME')["ARR_DELAY"].count(),
            "mean_departure_delay": df.groupby('ORIGIN_CITY_NAME')["DEP_DELAY"].mean(),
            "mean_arrival_delay": df.groupby('ORIGIN_CITY_NAME')["ARR_DELAY"].mean()
        }
    )
    return airport_df

    # Numeric feature set used for PCA + anomaly detection.
# (No carrier / distance / taxi / elapsed-time columns exist in this extract.)
FEATURE_COLS = ["DEP_DELAY", "ARR_DELAY", "ARR_DELAY_NEW", "DELAY_GAP", "ARR_HOUR", "DAY_OF_WEEK"]


def get_feature_matrix(d):
    cols = [c for c in FEATURE_COLS if c in d.columns]
    X = d[cols].astype(float)
    return X, cols


# ======================================================================
# SIDEBAR — DATA LOAD & CONTROLS
# ======================================================================
st.sidebar.title("\u2708\ufe0f Control Tower Settings")

# The dataset filename is fixed here since this dashboard is built for one
# specific file. Change this string if your CSV has a different name / path,
# then re-run `streamlit run dash.py` (no upload step needed).
DATA_FILE = r"C:\Users\steve\PycharmProjects\Anomaly-detection\data cleaning\Flights1_2019_1.csv"

try:
    raw_df = load_data(DATA_FILE)
except Exception as e:
    st.title("\u2708\ufe0f Flight Anomaly Control Tower")
    st.error(
        f"Could not load '{DATA_FILE}': {e}\n\n"
        f"Make sure this CSV is in the same folder as this script "
        f"(or update DATA_FILE near the top of the sidebar section)."
    )
    st.stop()

st.sidebar.success(f"Loaded: {DATA_FILE} ({len(raw_df):,} rows)")

required_cols = {"DEP_DELAY", "ARR_DELAY", "ARR_TIME", "DAY_OF_WEEK",
                  "ORIGIN_CITY_NAME", "DEST_CITY_NAME"}
missing = required_cols - set(raw_df.columns)
if missing:
    st.error(f"This file is missing expected columns: {missing}. "
             f"Columns found: {raw_df.columns.tolist()}")
    st.stop()

st.sidebar.markdown("---")

with st.sidebar.form("controls_form"):
    st.subheader("Settings")
    sample_n = st.slider(
        "Rows to use for modeling (subsample for speed)",
        min_value=1000, max_value=min(150000, max(2000, len(raw_df))),
        value=min(15000, len(raw_df)), step=1000,
        help="Lower = faster. One-Class SVM and LOF scale roughly O(n²), so this is the single biggest lever on runtime.",
    )

    dow_map = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    sel_dow = st.multiselect("Day(s) of week", list(dow_map.values()), default=list(dow_map.values()))
    sel_state = st.multiselect(
        "Destination state(s) (leave empty = all)",
        sorted(raw_df["DEST_STATE_ABR"].dropna().unique().tolist()), default=[]
    )

    contamination = st.slider("Expected anomaly fraction (contamination)", 0.01, 0.20, 0.05, 0.01)
    n_components = st.slider("PCA components (for modeling)", 2, 6, 3)

    submitted = st.form_submit_button("\u25B6 Apply & Run Analysis", use_container_width=True)

st.sidebar.caption(
    "Nothing recomputes until you click 'Apply & Run Analysis' above — "
    "this avoids retraining all 5 models every time you nudge a slider."
)

df, cancelled_by_dow = preprocess(raw_df, sample_n)

flight_path_df = transform_flightpath(df)

if len(df) < 50:
    st.error("Not enough clean rows after preprocessing. Check the file / column names.")
    st.stop()

X_raw, feature_cols = get_feature_matrix(df)
df = df.loc[X_raw.index].reset_index(drop=True)
X_raw = X_raw.reset_index(drop=True)

df["DOW_LABEL"] = df["DAY_OF_WEEK"].map(dow_map)
if sel_dow:
    mask = df["DOW_LABEL"].isin(sel_dow)
    df = df[mask].reset_index(drop=True)
    X_raw = X_raw.loc[mask[mask].index].reset_index(drop=True)

if sel_state:
    mask = df["DEST_STATE_ABR"].isin(sel_state)
    df = df[mask].reset_index(drop=True)
    X_raw = X_raw.loc[mask[mask].index].reset_index(drop=True)

n_components = min(n_components, len(feature_cols))

if len(df) < 50 or len(X_raw) < 50:
    st.error("Filters removed too much data — please relax filters.")
    st.stop()

# ======================================================================
# SCALING + PCA (dimension reduction)
# ======================================================================
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_raw)
#
# pca_full = PCA(n_components=min(len(feature_cols), 6), random_state=42)
# pca_full.fit(X_scaled)
# explained_var = pca_full.explained_variance_ratio_
#
# pca_model = PCA(n_components=n_components, random_state=42)
# X_pca = pca_model.fit_transform(X_scaled)
# pca_cols = [f"PC{i+1}" for i in range(n_components)]
# pca_df = pd.DataFrame(X_pca, columns=pca_cols)


# ======================================================================
# SCALING + PCA (dimension reduction) (Transformed Data)
# ======================================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(flight_path_df)

pca_full = PCA(n_components=6)
pca_full.fit(X_scaled)
explained_var = pca_full.explained_variance_ratio_

pca_model = PCA(n_components=6)
X_pca = pca_model.fit_transform(X_scaled)
pca_cols = [f"PC{i+1}" for i in range(6)]
pca_df = pd.DataFrame(X_pca, columns=pca_cols)


transform_scaled = scaler.fit_transform(X_scaled)

# ======================================================================
# ANOMALY DETECTION — 5 ALGORITHMS
# ======================================================================
# LOF and One-Class SVM are both roughly O(n^2) in the number of rows --
# at 150,000 rows that's ~22 billion pairwise computations, which is what
# was stalling the app. Isolation Forest, Elliptic Envelope, and DBSCAN all
# scale fine at this size, so only these two get capped: they TRAIN on a
# bounded random subsample but still SCORE every single row you selected.
MAX_TRAIN_N_FOR_QUADRATIC_MODELS = 12000


@st.cache_data(show_spinner=False)
def run_models(X_scaled_arr, contamination, seed=42, max_train_n=MAX_TRAIN_N_FOR_QUADRATIC_MODELS):
    results = {}
    n = X_scaled_arr.shape[0]

    iso = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
    iso_pred = iso.fit_predict(X_scaled_arr)
    iso_score = -iso.score_samples(X_scaled_arr)
    results["Isolation Forest"] = {"pred": iso_pred, "score": iso_score}

    # --- capped subsample used only for the two O(n^2) models ---
    if n > max_train_n:
        rng = np.random.default_rng(seed)
        train_idx = rng.choice(n, size=max_train_n, replace=False)
        X_train = X_scaled_arr[train_idx]
    else:
        X_train = X_scaled_arr

    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
    lof.fit(X_train)
    lof_pred = lof.predict(X_scaled_arr)
    lof_score = -lof.score_samples(X_scaled_arr)
    results["Local Outlier Factor"] = {"pred": lof_pred, "score": lof_score}

    ocsvm = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
    ocsvm.fit(X_train)
    ocsvm_pred = ocsvm.predict(X_scaled_arr)
    ocsvm_score = -ocsvm.decision_function(X_scaled_arr)
    results["One-Class SVM"] = {"pred": ocsvm_pred, "score": ocsvm_score}

    try:
        ee = EllipticEnvelope(contamination=contamination, random_state=seed)
        ee_pred = ee.fit_predict(X_scaled_arr)
        ee_score = -ee.score_samples(X_scaled_arr)
        results["Elliptic Envelope"] = {"pred": ee_pred, "score": ee_score}
    except Exception:
        pass

    db = DBSCAN(eps=1.2, min_samples=10)
    db_labels = db.fit_predict(X_scaled_arr)
    db_pred = np.where(db_labels == -1, -1, 1)
    db_score = (db_labels == -1).astype(float)
    results["DBSCAN"] = {"pred": db_pred, "score": db_score}

    return results


with st.spinner("Running anomaly detection models..."):
    model_results = run_models(X_scaled, contamination)

if len(X_scaled) > MAX_TRAIN_N_FOR_QUADRATIC_MODELS:
    st.info(
        f"\u2139\ufe0f You selected **{len(X_scaled):,}** rows. Isolation Forest, Elliptic Envelope, "
        f"and DBSCAN scale fine at this size and ran on all of them. Local Outlier Factor and "
        f"One-Class SVM are O(n\u00b2) and were **trained on a random {MAX_TRAIN_N_FOR_QUADRATIC_MODELS:,}-row "
        f"subsample** instead — but they still scored and flagged every one of your {len(X_scaled):,} rows."
    )

algo_names = list(model_results.keys())
for name, r in model_results.items():
    flight_path_df[f"flag_{name}"] = (r["pred"] == -1)
    s = r["score"]
    s_norm = (s - s.min()) / (s.max() - s.min() + 1e-9)
    flight_path_df[f"score_{name}"] = s_norm

flight_path_df["consensus_votes"] = flight_path_df[[f"flag_{n}" for n in algo_names]].sum(axis=1)
flight_path_df["consensus_anomaly"] = flight_path_df["consensus_votes"] >= 2

# ======================================================================
# TABS
# ======================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["\U0001F4CA Explore & Visualize", "\U0001F53B Dimension Reduction",
     "\U0001F6A8 Anomaly Detection", "\u2705 Validation", "\U0001F50D Flight Drill-Down"]
)

# ----------------------------------------------------------------------
# TAB 1 — EDA (all 3D where it adds real information)
# ----------------------------------------------------------------------
with tab1:
    st.header("Explore & Visualize the Dataset")
    st.caption(
        "This extract has no carrier, distance, or taxi-time columns — visuals below are "
        "built around delay, timing, day-of-week, and route fields actually present in the data."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flights (analyzed)", f"{len(df):,}")
    c2.metric("Avg Dep Delay (min)", f"{df['DEP_DELAY'].mean():.1f}")
    c3.metric("Avg Arr Delay (min)", f"{df['ARR_DELAY'].mean():.1f}")
    c4.metric("% Delayed 15+ min (ARR_DEL15)", f"{df['ARR_DEL15'].mean()*100:.1f}%")

    st.subheader("Delay Distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="DEP_DELAY", nbins=80, title="Departure Delay Distribution")
        fig.add_vline(x=df["DEP_DELAY"].mean(), line_dash="dash", line_color="red",
                       annotation_text="mean")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x="ARR_DELAY", nbins=80, title="Arrival Delay Distribution",
                            color_discrete_sequence=["orange"])
        fig.add_vline(x=df["ARR_DELAY"].mean(), line_dash="dash", line_color="red",
                       annotation_text="mean")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.pie(
            df, names=df["ARR_DEL15"].map({0: "On-time / <15min", 1: "Delayed 15+ min"}),
            title="Share of Flights Delayed 15+ Minutes",
            color_discrete_sequence=["#2ca02c", "#d62728"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        state_delay = df.groupby("DEST_STATE_ABR")["ARR_DELAY"].mean().sort_values(ascending=False).head(15)
        fig = px.bar(state_delay, orientation="h", title="Top 15 Destination States by Avg Arrival Delay",
                     labels={"value": "Avg Arr Delay (min)", "DEST_STATE_ABR": "State"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily Trend Across January 2019")
    daily = df.groupby(df["FL_DATE"].dt.date).agg(
        avg_dep_delay=("DEP_DELAY", "mean"),
        avg_arr_delay=("ARR_DELAY", "mean"),
        flights=("ARR_DELAY", "size"),
    ).reset_index()
    col5, col6 = st.columns(2)
    with col5:
        fig = px.line(daily, x="FL_DATE", y=["avg_dep_delay", "avg_arr_delay"], markers=True,
                       title="Average Delay by Day (January 2019)",
                       labels={"value": "Avg Delay (min)", "FL_DATE": "Date", "variable": "Metric"})
        st.plotly_chart(fig, use_container_width=True)
    with col6:
        fig = px.bar(daily, x="FL_DATE", y="flights", title="Flight Volume by Day (sampled)",
                     labels={"flights": "Number of Flights", "FL_DATE": "Date"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Departure Delay vs Arrival Delay (2D)")
    st.caption("A precise 2D read of the same relationship shown in 3D below — useful for reading exact values.")
    fig = px.scatter(
        df, x="DEP_DELAY", y="ARR_DELAY", color="DOW_LABEL", opacity=0.4,
        title="Departure Delay vs Arrival Delay, colored by Day of Week",
        trendline="ols",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Likely Cancelled / Diverted Flights")
    st.caption(
        "This extract has no explicit CANCELLED column — rows missing both DEP_DELAY and "
        "ARR_DELAY are treated as likely cancelled/diverted flights, tallied here by day of week "
        "(computed before those rows were dropped for modeling)."
    )
    fig = px.bar(
        (cancelled_by_dow * 100).rename(index=dow_map),
        title="% of Flights Likely Cancelled/Diverted, by Day of Week",
        labels={"value": "% Cancelled/Diverted", "index": "Day of Week"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 3D VIZ #1 — Delay Relationships
    # ------------------------------------------------------------------
    st.subheader("3D Delay Relationships")
    st.caption(
        "Each point is a flight: Departure Delay x Arrival Delay x Hour of Arrival, colored by "
        "day of week. Rotate to spot flights that are extreme on more than one axis at once. "
        "(Capped at 5,000 points for smooth rotation — statistics elsewhere use the full sample.)"
    )
    render_df1 = df.sample(min(5000, len(df)), random_state=42)
    fig = px.scatter_3d(
        render_df1, x="DEP_DELAY", y="ARR_DELAY", z="ARR_HOUR",
        color="DOW_LABEL",
        opacity=0.5,
        title="Departure Delay vs Arrival Delay vs Hour of Arrival",
        labels={"DEP_DELAY": "Dep Delay (min)", "ARR_DELAY": "Arr Delay (min)", "ARR_HOUR": "Arrival Hour"},
    )
    fig.update_layout(scene=dict(aspectmode="cube"), margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 3D VIZ #2 — Delay terrain: hour of day x day of week
    # ------------------------------------------------------------------
    st.subheader("3D Delay Terrain — Hour of Arrival vs Day of Week")
    st.caption("Height/color = average arrival delay. Peaks show the worst hour/day combinations.")
    pivot = df.pivot_table(index="DAY_OF_WEEK", columns="ARR_HOUR", values="ARR_DELAY", aggfunc="mean")
    pivot = pivot.reindex(index=range(1, 8), columns=range(24)).interpolate(axis=1, limit_direction="both")
    fig = go.Figure(data=[go.Surface(
        z=pivot.values, x=pivot.columns, y=[dow_map[i] for i in pivot.index],
        colorscale="RdYlGn_r", colorbar=dict(title="Avg Arr Delay (min)"),
    )])
    fig.update_layout(
        title="Average Arrival Delay by Hour and Day of Week",
        scene=dict(xaxis_title="Hour of Arrival", yaxis_title="Day of Week", zaxis_title="Avg Arr Delay (min)"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 3D VIZ #3 — Route volume bubble (origin city x dest city x count)
    # ------------------------------------------------------------------
    st.subheader("3D Route Volume")
    st.caption("Origin city and destination city on the base plane, bubble size/height = flight count.")
    route_counts = df.groupby(["ORIGIN_CITY_NAME", "DEST_CITY_NAME"]).size().reset_index(name="COUNT")
    route_counts = route_counts.sort_values("COUNT", ascending=False).head(40)
    fig = px.scatter_3d(
        route_counts, x="ORIGIN_CITY_NAME", y="DEST_CITY_NAME", z="COUNT",
        size="COUNT", color="COUNT", color_continuous_scale="Turbo",
        size_max=30, opacity=0.8,
        title="Top 40 Routes by Volume",
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 3D VIZ #4 — Delay distribution shape by day of week (ridgeline)
    # ------------------------------------------------------------------
    st.subheader("3D Delay Distribution Shape by Day of Week")
    st.caption("A 3D ridgeline: traces the arrival-delay histogram shape for each day of week.")
    bins = np.linspace(df["ARR_DELAY"].quantile(0.01), df["ARR_DELAY"].quantile(0.99), 40)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    z_matrix = []
    dow_order = list(range(1, 8))
    for dow in dow_order:
        vals = df.loc[df["DAY_OF_WEEK"] == dow, "ARR_DELAY"]
        hist, _ = np.histogram(vals, bins=bins, density=True)
        z_matrix.append(hist)
    fig = go.Figure(data=[go.Surface(
        z=np.array(z_matrix), x=bin_centers, y=[dow_map[d] for d in dow_order],
        colorscale="Viridis", colorbar=dict(title="Density"),
    )])
    fig.update_layout(
        title="Arrival Delay Distribution Shape, per Day of Week",
        scene=dict(xaxis_title="Arrival Delay (min)", yaxis_title="Day of Week", zaxis_title="Density"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Correlation heatmap + boxplots — deliberately kept 2D
    # ------------------------------------------------------------------
    st.subheader("Correlation Heatmap (numeric features)")
    st.caption("Kept 2D on purpose — a heatmap already encodes the relationship via color + grid position.")
    corr = X_raw.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title="Feature Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Boxplots — Outlier Spotting per Feature")
    st.caption("Kept 2D — boxplots are a 1D summary with nothing spatial to gain from a 3rd axis.")
    col_a, col_b = st.columns(2)
    with col_a:
        feat_for_box = st.selectbox("Select feature (boxplot)", feature_cols, key="box_feat")
        fig = px.box(X_raw, y=feat_for_box, points="outliers", title=f"Boxplot of {feat_for_box}")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        feat_for_violin = st.selectbox("Select feature (violin, split by day type)", feature_cols, key="violin_feat")
        violin_df = X_raw.copy()
        violin_df["DOW_LABEL"] = df["DOW_LABEL"].values
        fig = px.violin(violin_df, y=feat_for_violin, x="DOW_LABEL", box=True, points=False,
                         title=f"Distribution of {feat_for_violin} by Day of Week")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pairwise Feature Relationships (Scatter Matrix)")
    st.caption(
        "Every feature plotted against every other — a quick way to spot which pairs separate cleanly. "
        "(Capped at 3,000 points — this chart draws ~15 panels at once, so it's the heaviest one to render.)"
    )
    splom_df = X_raw.copy()
    splom_df["DOW_LABEL"] = df["DOW_LABEL"].values
    splom_df = splom_df.sample(min(3000, len(splom_df)), random_state=42)
    fig = px.scatter_matrix(
        splom_df, dimensions=feature_cols, color="DOW_LABEL", opacity=0.35,
        title="Scatter Matrix of All Numeric Features",
    )
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# TAB 2 — DIMENSION REDUCTION
# ----------------------------------------------------------------------
with tab2:
    st.header("Dimension Reduction (PCA)")
    st.markdown(
        f"Feature space has **{len(feature_cols)}** numeric dimensions: "
        f"`{', '.join(feature_cols)}`. PCA compresses this into **{n_components}** components "
        "for visualization and to stabilize the anomaly detection models."
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            x=[f"PC{i+1}" for i in range(len(explained_var))], y=explained_var,
            title="Explained Variance per Principal Component",
            labels={"x": "Component", "y": "Explained Variance Ratio"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        cum_var = np.cumsum(explained_var)
        fig = px.line(x=[f"PC{i+1}" for i in range(len(cum_var))], y=cum_var, markers=True,
                      title="Cumulative Explained Variance")
        fig.add_hline(y=0.9, line_dash="dash", line_color="red", annotation_text="90% threshold")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"Using **{n_components}** components explains **{cum_var[n_components-1]*100:.1f}%** of total variance.")

    st.subheader("Flights Projected onto Principal Components")
    plot_df = pca_df.copy()
    plot_df["color_val"] = df["ARR_DELAY"].values
    if n_components >= 3:
        fig = px.scatter_3d(
            plot_df, x="PC1", y="PC2", z="PC3", color="color_val",
            color_continuous_scale="RdYlGn_r", title="3D PCA Projection (colored by arrival delay)",
            opacity=0.6,
        )
    else:
        fig = px.scatter(
            plot_df, x="PC1", y="PC2", color="color_val",
            color_continuous_scale="RdYlGn_r", title="2D PCA Projection (colored by arrival delay)",
            opacity=0.6,
        )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("2D PCA Biplot (PC1 vs PC2) with Loading Vectors")
    st.caption("Arrows show how much each original feature contributes to PC1/PC2 and in which direction — useful for interpreting what each axis 'means'.")
    biplot_df = pca_df.copy()
    biplot_df["DOW_LABEL"] = df["DOW_LABEL"].values
    fig = px.scatter(biplot_df, x="PC1", y="PC2", color="DOW_LABEL", opacity=0.35,
                      title="PCA Biplot: Flights (points) + Feature Loadings (arrows)")
    scale = 3 * biplot_df[["PC1", "PC2"]].abs().quantile(0.95).max()
    loadings_2d = pd.DataFrame(pca_model.components_[:2].T, index=feature_cols, columns=["PC1", "PC2"])
    for feat, row in loadings_2d.iterrows():
        fig.add_shape(type="line", x0=0, y0=0, x1=row["PC1"] * scale, y1=row["PC2"] * scale,
                       line=dict(color="black", width=2))
        fig.add_annotation(x=row["PC1"] * scale, y=row["PC2"] * scale, text=feat,
                            showarrow=False, font=dict(color="black", size=11))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("PCA Projection Split by Day of Week")
    fig = px.scatter(biplot_df, x="PC1", y="PC2", facet_col="DOW_LABEL", facet_col_wrap=4,
                      opacity=0.4, title="PC1 vs PC2, Faceted by Day of Week", height=500)
    st.plotly_chart(fig, use_container_width=True)


    loadings = pd.DataFrame(pca_model.components_.T, index=feature_cols, columns=pca_cols)
    fig = px.imshow(loadings, text_auto=".2f", color_continuous_scale="RdBu_r",
                     title="Loadings: how each original feature maps into PCA space")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# TAB 3 — ANOMALY DETECTION
# ----------------------------------------------------------------------
with tab3:
    st.header("Anomaly Detection — 5 Algorithms")
    st.markdown(
        f"Contamination / expected anomaly rate = **{contamination:.0%}**. "
        "A flight is flagged (\U0001F6A8) if the model considers it an outlier."
    )

    counts = {name: int(df[f"flag_{name}"].sum()) for name in algo_names}
    cols = st.columns(len(algo_names))
    for c, name in zip(cols, algo_names):
        c.metric(name, f"{counts[name]:,} flagged", f"{counts[name]/len(df)*100:.1f}%")

    st.subheader("Where the anomalies sit in PCA space")
    algo_pick = st.selectbox("Choose algorithm to visualize", algo_names)
    plot_df2 = pca_df.copy()
    plot_df2["Anomaly"] = df[f"flag_{algo_pick}"].map({True: "Anomaly", False: "Normal"})
    plot_df2["Score"] = df[f"score_{algo_pick}"]
    fig = px.scatter_3d(
        plot_df2, x="PC1", y="PC2", z="PC3" if n_components >= 3 else "PC1",
        color="Anomaly", color_discrete_map={"Anomaly": "red", "Normal": "lightblue"},
        hover_data=["Score"], title=f"{algo_pick}: Anomalies in PCA Space", opacity=0.6,
    ) if n_components >= 3 else px.scatter(
        plot_df2, x="PC1", y="PC2", color="Anomaly",
        color_discrete_map={"Anomaly": "red", "Normal": "lightblue"},
        hover_data=["Score"], title=f"{algo_pick}: Anomalies in PCA Space", opacity=0.7,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Anomaly Score Distribution")
    st.caption("Where the flagging threshold falls for the selected algorithm — the tail past the red line is what gets flagged.")
    score_series = df[f"score_{algo_pick}"]
    thresh = score_series[df[f"flag_{algo_pick}"]].min() if df[f"flag_{algo_pick}"].any() else score_series.max()
    fig = px.histogram(df, x=f"score_{algo_pick}", nbins=60, title=f"{algo_pick}: Score Distribution")
    fig.add_vline(x=thresh, line_dash="dash", line_color="red", annotation_text="flag threshold")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Anomaly Rate by Hour and Day of Week")
    st.caption("Where in the week are anomalies concentrated? Darker cells = higher share of flights flagged.")
    rate_pivot = df.pivot_table(index="DAY_OF_WEEK", columns="ARR_HOUR",
                                 values=f"flag_{algo_pick}", aggfunc="mean")
    rate_pivot = rate_pivot.reindex(index=range(1, 8), columns=range(24)).fillna(0)
    fig = px.imshow(
        rate_pivot, y=[dow_map[i] for i in rate_pivot.index], x=rate_pivot.columns,
        color_continuous_scale="Reds", title=f"{algo_pick}: Anomaly Rate by Hour x Day of Week",
        labels={"x": "Hour of Arrival", "y": "Day of Week", "color": "Anomaly Rate"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Consensus View — Voting Across All Algorithms")
    vote_counts = df["consensus_votes"].value_counts().sort_index()
    fig = px.bar(vote_counts, title="Number of Flights by Vote Count",
                 labels={"index": "Number of algorithms flagging flight", "value": "Number of flights"})
    st.plotly_chart(fig, use_container_width=True)

    plot_df3 = pca_df.copy()
    plot_df3["Votes"] = df["consensus_votes"]
    if n_components >= 3:
        fig = px.scatter_3d(
            plot_df3, x="PC1", y="PC2", z="PC3", color="Votes",
            color_continuous_scale="Reds", title="Consensus Anomaly Score in PCA Space", opacity=0.6,
        )
    else:
        fig = px.scatter(
            plot_df3, x="PC1", y="PC2", color="Votes",
            color_continuous_scale="Reds", title="Consensus Anomaly Score in PCA Space", opacity=0.7,
        )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Consensus Anomalies")
    show_cols = ["FL_DATE", "ORIGIN_CITY_NAME", "DEST_CITY_NAME", "DEP_DELAY", "ARR_DELAY",
                 "DOW_LABEL", "consensus_votes"]
    top_anom = df.sort_values("consensus_votes", ascending=False)[show_cols].head(20)
    st.dataframe(top_anom, use_container_width=True)

# ----------------------------------------------------------------------
# TAB 4 — VALIDATION
# ----------------------------------------------------------------------
with tab4:
    st.header("Validating the Anomaly Detection Results")
    st.markdown(
        "No single algorithm is ground truth, so we validate via agreement between methods. "
        "Consistent overlap across very different algorithms (density-based, tree-based, "
        "distance-based, covariance-based) is stronger evidence flagged flights are genuinely "
        "anomalous, not an artifact of one model's assumptions."
    )

    st.subheader("Pairwise Agreement (Jaccard Similarity of Flagged Sets)")
    jaccard = pd.DataFrame(index=algo_names, columns=algo_names, dtype=float)
    for a in algo_names:
        for b in algo_names:
            set_a = set(df.index[df[f"flag_{a}"]])
            set_b = set(df.index[df[f"flag_{b}"]])
            union = len(set_a | set_b)
            inter = len(set_a & set_b)
            jaccard.loc[a, b] = inter / union if union > 0 else 0.0
    fig = px.imshow(jaccard.astype(float), text_auto=".2f", color_continuous_scale="Viridis",
                     title="Jaccard Similarity Between Algorithms' Flagged Flights")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rank Correlation of Anomaly Scores (Spearman)")
    score_cols = [f"score_{n}" for n in algo_names]
    spearman_corr = df[score_cols].corr(method="spearman")
    spearman_corr.columns = algo_names
    spearman_corr.index = algo_names
    fig = px.imshow(spearman_corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title="Spearman Correlation Between Algorithm Anomaly Scores")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Direct Score Comparison Between Two Algorithms")
    st.caption("If two very different algorithms agree, points cluster along the diagonal at the top-right — strong evidence those flights are genuinely anomalous.")
    col_x, col_y = st.columns(2)
    with col_x:
        algo_x = st.selectbox("Algorithm A (x-axis)", algo_names, index=0, key="valx")
    with col_y:
        algo_y = st.selectbox("Algorithm B (y-axis)", algo_names, index=min(1, len(algo_names)-1), key="valy")
    comp_scores_df = pd.DataFrame({
        algo_x: df[f"score_{algo_x}"], algo_y: df[f"score_{algo_y}"],
        "Consensus": df["consensus_anomaly"].map({True: "Consensus Anomaly", False: "Normal"}),
    })
    fig = px.scatter(comp_scores_df, x=algo_x, y=algo_y, color="Consensus",
                      color_discrete_map={"Consensus Anomaly": "red", "Normal": "lightblue"},
                      opacity=0.5, title=f"{algo_x} Score vs {algo_y} Score")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sanity Check — Do flagged flights actually look extreme?")
    compare_cols = ["DEP_DELAY", "ARR_DELAY", "DELAY_GAP", "ARR_DEL15"]
    comp = df.groupby("consensus_anomaly")[compare_cols].mean().rename(
        index={True: "Consensus Anomaly", False: "Normal"}
    )
    st.dataframe(comp.style.format("{:.2f}"), use_container_width=True)

    fig = go.Figure()
    for c in compare_cols:
        fig.add_trace(go.Bar(name=c, x=["Normal", "Consensus Anomaly"],
                              y=[comp.loc["Normal", c], comp.loc["Consensus Anomaly", c]]))
    fig.update_layout(barmode="group", title="Average Feature Values: Normal vs Consensus Anomalies")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Silhouette-style separation check (optional statistic)")
    try:
        from sklearn.metrics import silhouette_score
        labels_bin = df["consensus_anomaly"].astype(int).values
        if 0 < labels_bin.sum() < len(labels_bin):
            sil = silhouette_score(X_scaled, labels_bin)
            st.metric("Silhouette score (normal vs consensus anomaly)", f"{sil:.3f}")
            st.caption("Closer to +1 means the two groups are well separated in feature space.")
    except Exception as e:
        st.caption(f"Silhouette score unavailable: {e}")

# ----------------------------------------------------------------------
# TAB 5 — FLIGHT DRILL-DOWN
# ----------------------------------------------------------------------
with tab5:
    st.header("Flight Drill-Down: Why Was This Flight Flagged?")

    ranked = df.sort_values("consensus_votes", ascending=False).reset_index()
    ranked = ranked.rename(columns={"index": "row_id"})

    def make_label(row):
        d = row["FL_DATE"].date() if pd.notna(row["FL_DATE"]) else "?"
        return (f"{d} | {row['ORIGIN_CITY_NAME']} \u2192 {row['DEST_CITY_NAME']} "
                f"(votes={row['consensus_votes']})")

    ranked["label"] = ranked.apply(make_label, axis=1)

    st.markdown("Pick a flight below (defaults to the most suspicious one) to inspect it in detail.")
    choice_label = st.selectbox("Select a flight", ranked["label"].tolist())
    sel_row = ranked[ranked["label"] == choice_label].iloc[0]
    row_id = int(sel_row["row_id"])

    flight_row = df.loc[row_id]
    x_row = X_raw.loc[row_id]

    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithms flagging this flight", f"{int(flight_row['consensus_votes'])} / {len(algo_names)}")
    c2.metric("Departure Delay", f"{flight_row['DEP_DELAY']:.0f} min")
    c3.metric("Arrival Delay", f"{flight_row['ARR_DELAY']:.0f} min")

    st.subheader("Raw Flight Record")
    show_fields = ["FL_DATE", "DOW_LABEL", "ORIGIN_CITY_NAME", "DEST_CITY_NAME", "DEST_STATE_ABR",
                   "DEP_DELAY", "ARR_TIME", "ARR_DELAY", "ARR_DEL15"]
    st.dataframe(flight_row[show_fields].to_frame().T, use_container_width=True)

    st.subheader("Per-Algorithm Verdict")
    verdict_df = pd.DataFrame({
        "Algorithm": algo_names,
        "Flagged?": [bool(flight_row[f"flag_{n}"]) for n in algo_names],
        "Anomaly Score (0-1, normalized)": [round(float(flight_row[f"score_{n}"]), 3) for n in algo_names],
    })
    st.dataframe(verdict_df, use_container_width=True)

    st.subheader("Why is this flight anomalous? (Feature Deviation Explanation)")
    means = X_raw.mean()
    stds = X_raw.std().replace(0, 1)
    z = (x_row - means) / stds
    z_sorted = z.reindex(z.abs().sort_values(ascending=False).index)

    explain_df = pd.DataFrame({
        "Feature": z_sorted.index,
        "This flight's value": [x_row[f] for f in z_sorted.index],
        "Dataset average": [means[f] for f in z_sorted.index],
        "Z-score (std devs from average)": z_sorted.values,
    })
    st.dataframe(explain_df.style.format({
        "This flight's value": "{:.1f}",
        "Dataset average": "{:.1f}",
        "Z-score (std devs from average)": "{:.2f}",
    }), use_container_width=True)

    fig = px.bar(
        explain_df, x="Z-score (std devs from average)", y="Feature",
        orientation="h", color="Z-score (std devs from average)",
        color_continuous_scale="RdBu_r",
        title="Top Contributing Features (largest deviation from typical flight)",
    )
    st.plotly_chart(fig, use_container_width=True)

    top_feat = z_sorted.index[0]
    top_z = z_sorted.iloc[0]
    direction = "higher" if top_z > 0 else "lower"
    st.info(
        f"**Plain-language summary:** this flight is most unusual because its "
        f"**{top_feat}** ({x_row[top_feat]:.1f}) is **{abs(top_z):.1f} standard deviations {direction}** "
        f"than the typical flight in this dataset (average = {means[top_feat]:.1f}). "
        f"It was flagged by **{int(flight_row['consensus_votes'])} out of {len(algo_names)}** algorithms."
    )

    st.subheader("Where this flight sits relative to everyone else (PCA space)")
    highlight_df = pca_df.copy()
    highlight_df["Type"] = "Other flights"
    highlight_df.loc[row_id, "Type"] = "Selected flight"
    if n_components >= 3:
        fig = px.scatter_3d(
            highlight_df, x="PC1", y="PC2", z="PC3", color="Type",
            color_discrete_map={"Other flights": "lightgray", "Selected flight": "red"},
            title="Selected Flight Highlighted in PCA Space",
        )
    else:
        fig = px.scatter(
            highlight_df, x="PC1", y="PC2", color="Type",
            color_discrete_map={"Other flights": "lightgray", "Selected flight": "red"},
            title="Selected Flight Highlighted in PCA Space",
        )
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Flight Anomaly Control Tower — RITA/BTS On-Time Performance Data (Jan 2019)")