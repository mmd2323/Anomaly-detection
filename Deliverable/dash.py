"""
Flight Anomaly Control Tower
=====================================
Interactive Streamlit dashboard for detecting anomalous flights in the
RITA / BTS Reporting Carrier On-Time Performance dataset.

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
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="Flight Anomaly Control Tower",
    layout="wide",
    page_icon="\u2708\ufe0f",
)

# ======================================================================
# COLUMN MAPPING (handles different RITA/BTS export schemas gracefully)
# ======================================================================
CANDIDATE_COLUMNS = {
    "date": ["FL_DATE", "FlightDate"],
    "carrier": ["OP_UNIQUE_CARRIER", "OP_CARRIER", "Reporting_Airline", "UNIQUE_CARRIER"],
    "flight_num": ["OP_CARRIER_FL_NUM", "Flight_Number_Reporting_Airline"],
    "origin": ["ORIGIN"],
    "dest": ["DEST"],
    "crs_dep": ["CRS_DEP_TIME"],
    "dep_time": ["DEP_TIME"],
    "dep_delay": ["DEP_DELAY"],
    "taxi_out": ["TAXI_OUT"],
    "wheels_off": ["WHEELS_OFF"],
    "wheels_on": ["WHEELS_ON"],
    "taxi_in": ["TAXI_IN"],
    "crs_arr": ["CRS_ARR_TIME"],
    "arr_time": ["ARR_TIME"],
    "arr_delay": ["ARR_DELAY"],
    "cancelled": ["CANCELLED"],
    "diverted": ["DIVERTED"],
    "crs_elapsed": ["CRS_ELAPSED_TIME"],
    "act_elapsed": ["ACTUAL_ELAPSED_TIME"],
    "air_time": ["AIR_TIME"],
    "distance": ["DISTANCE"],
    "carrier_delay": ["CARRIER_DELAY"],
    "weather_delay": ["WEATHER_DELAY"],
    "nas_delay": ["NAS_DELAY"],
    "security_delay": ["SECURITY_DELAY"],
    "late_aircraft_delay": ["LATE_AIRCRAFT_DELAY"],
}

NUMERIC_FEATURE_KEYS = [
    "dep_delay", "arr_delay", "taxi_out", "taxi_in",
    "air_time", "distance", "crs_elapsed", "act_elapsed",
]


def map_columns(df):
    colmap = {}
    for key, options in CANDIDATE_COLUMNS.items():
        for opt in options:
            if opt in df.columns:
                colmap[key] = opt
                break
    return colmap


# ======================================================================
# DATA LOADING / CLEANING
# ======================================================================
@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file, low_memory=False)
    return df


@st.cache_data(show_spinner=False)
def preprocess(df, colmap, sample_n, random_state=42):
    d = df.copy()

    if "cancelled" in colmap:
        d = d[d[colmap["cancelled"]] == 0]
    if "diverted" in colmap:
        d = d[d[colmap["diverted"]] == 0]

    feature_cols = [colmap[k] for k in NUMERIC_FEATURE_KEYS if k in colmap]
    d = d.dropna(subset=feature_cols)

    for c in feature_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=feature_cols)

    # Derived features
    if "act_elapsed" in colmap and "crs_elapsed" in colmap:
        d["ELAPSED_DIFF"] = d[colmap["act_elapsed"]] - d[colmap["crs_elapsed"]]
    if "distance" in colmap and "air_time" in colmap:
        with np.errstate(divide="ignore", invalid="ignore"):
            d["SPEED_MPH"] = d[colmap["distance"]] / (d[colmap["air_time"]] / 60.0)
        d["SPEED_MPH"] = d["SPEED_MPH"].replace([np.inf, -np.inf], np.nan)

    d = d.dropna(subset=[c for c in ["ELAPSED_DIFF", "SPEED_MPH"] if c in d.columns])

    if len(d) > sample_n:
        d = d.sample(sample_n, random_state=random_state)

    return d.reset_index(drop=True)


def get_feature_matrix(d, colmap):
    cols = [colmap[k] for k in NUMERIC_FEATURE_KEYS if k in colmap]
    extra = [c for c in ["ELAPSED_DIFF", "SPEED_MPH"] if c in d.columns]
    cols = cols + extra
    X = d[cols].astype(float)
    return X, cols


# ======================================================================
# SYNTHETIC DATA GENERATOR (used when no real CSV is available yet)
# ======================================================================
@st.cache_data(show_spinner=False)
def generate_synthetic_flights(n=15000, seed=42):
    rng = np.random.default_rng(seed)

    carriers = ["AA", "DL", "UA", "WN", "AS", "B6", "NK", "F9"]
    airports = ["ATL", "ORD", "DFW", "DEN", "LAX", "JFK", "SFO", "SEA",
                "MIA", "PHX", "IAH", "BOS", "MSP", "DTW", "CLT"]

    n_normal = int(n * 0.95)
    n_anom = n - n_normal

    def make_block(k, anomalous=False):
        origin = rng.choice(airports, size=k)
        dest = rng.choice(airports, size=k)
        same = origin == dest
        dest[same] = rng.choice(airports, size=same.sum())

        distance = rng.uniform(150, 2800, size=k)
        crs_elapsed = distance / rng.uniform(6.5, 8.0, size=k) + rng.normal(20, 5, size=k)
        crs_elapsed = np.clip(crs_elapsed, 35, None)

        if not anomalous:
            dep_delay = rng.normal(8, 15, size=k)
            arr_delay = dep_delay + rng.normal(-2, 12, size=k)
            taxi_out = np.clip(rng.normal(16, 6, size=k), 3, None)
            taxi_in = np.clip(rng.normal(7, 3, size=k), 1, None)
            act_elapsed = crs_elapsed + rng.normal(0, 8, size=k)
        else:
            # inject extreme / anomalous behavior
            kind = rng.integers(0, 4, size=k)
            dep_delay = np.where(kind == 0, rng.uniform(180, 600, size=k), rng.normal(8, 15, size=k))
            taxi_out = np.where(kind == 1, rng.uniform(90, 200, size=k), np.clip(rng.normal(16, 6, size=k), 3, None))
            taxi_in = np.where(kind == 2, rng.uniform(60, 150, size=k), np.clip(rng.normal(7, 3, size=k), 1, None))
            act_elapsed = np.where(kind == 3, crs_elapsed * rng.uniform(2.0, 3.5, size=k), crs_elapsed + rng.normal(0, 8, size=k))
            arr_delay = dep_delay + rng.normal(0, 20, size=k) + np.where(kind == 3, 120, 0)

        air_time = np.clip(act_elapsed - taxi_out - taxi_in, 15, None)

        dates = pd.to_datetime("2019-01-01") + pd.to_timedelta(rng.integers(0, 31, size=k), unit="D")
        crs_dep = rng.integers(0, 24, size=k) * 100 + rng.integers(0, 60, size=k)

        return pd.DataFrame({
            "FL_DATE": dates.astype(str),
            "OP_UNIQUE_CARRIER": rng.choice(carriers, size=k),
            "OP_CARRIER_FL_NUM": rng.integers(100, 9999, size=k),
            "ORIGIN": origin,
            "DEST": dest,
            "CRS_DEP_TIME": crs_dep,
            "DEP_DELAY": dep_delay,
            "TAXI_OUT": taxi_out,
            "TAXI_IN": taxi_in,
            "CRS_ARR_TIME": (crs_dep + crs_elapsed.astype(int)) % 2400,
            "ARR_DELAY": arr_delay,
            "CANCELLED": 0,
            "DIVERTED": 0,
            "CRS_ELAPSED_TIME": crs_elapsed,
            "ACTUAL_ELAPSED_TIME": act_elapsed,
            "AIR_TIME": air_time,
            "DISTANCE": distance,
        })

    df_normal = make_block(n_normal, anomalous=False)
    df_anom = make_block(n_anom, anomalous=True)
    full = pd.concat([df_normal, df_anom], ignore_index=True)
    return full.sample(frac=1, random_state=seed).reset_index(drop=True)


# ======================================================================
# SIDEBAR — DATA LOAD & CONTROLS
# ======================================================================
st.sidebar.title("\u2708\ufe0f Control Tower Settings")

use_synthetic = st.sidebar.checkbox(
    "\U0001F9EA Use synthetic demo data (no CSV needed)", value=True,
    help="Generates realistic fake flight records with built-in anomalies so you can test the dashboard before your real RITA file is ready."
)

uploaded_file = None
default_path = ""
if not use_synthetic:
    uploaded_file = st.sidebar.file_uploader("Upload RITA/BTS CSV (e.g. flights1_2019_1.csv)", type=["csv"])
    default_path = st.sidebar.text_input("...or local file path", value="flights1_2019_1.csv")

if use_synthetic:
    n_synth = st.sidebar.slider("Synthetic flights to generate", 2000, 50000, 15000, 1000)
    raw_df = generate_synthetic_flights(n=n_synth)
    st.sidebar.success(f"Using synthetic data: {len(raw_df):,} generated flights")
else:
    data_source = uploaded_file if uploaded_file is not None else default_path
    if not data_source:
        st.title("\u2708\ufe0f Flight Anomaly Control Tower")
        st.info("Upload a CSV file or provide a local file path in the sidebar to begin — or check "
                 "'Use synthetic demo data' to explore the dashboard right away.")
        st.stop()
    try:
        raw_df = load_data(data_source)
    except Exception as e:
        st.error(f"Could not load file: {e}")
        st.stop()

colmap = map_columns(raw_df)

st.sidebar.markdown("---")
sample_n = st.sidebar.slider(
    "Rows to use for modeling (subsample for speed)",
    min_value=1000, max_value=min(100000, max(2000, len(raw_df))),
    value=min(20000, len(raw_df)), step=1000,
)

df = preprocess(raw_df, colmap, sample_n)

if len(df) < 50:
    st.error("Not enough clean rows after preprocessing. Check the file / column names.")
    st.stop()

X_raw, feature_cols = get_feature_matrix(df, colmap)
df = df.loc[X_raw.index].reset_index(drop=True)
X_raw = X_raw.reset_index(drop=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

if "carrier" in colmap:
    carriers = sorted(df[colmap["carrier"]].dropna().unique().tolist())
    sel_carriers = st.sidebar.multiselect("Carrier(s)", carriers, default=carriers)
    if sel_carriers:
        mask = df[colmap["carrier"]].isin(sel_carriers)
        df = df[mask].reset_index(drop=True)
        X_raw = X_raw[mask.values].reset_index(drop=True) if len(mask) == len(X_raw) else X_raw

if "origin" in colmap:
    origins = sorted(df[colmap["origin"]].dropna().unique().tolist())
    sel_origin = st.sidebar.multiselect("Origin airport(s)", origins, default=[])
    if sel_origin:
        mask = df[colmap["origin"]].isin(sel_origin)
        df = df[mask].reset_index(drop=True)
        X_raw = X_raw.loc[mask[mask].index].reset_index(drop=True)

st.sidebar.markdown("---")
contamination = st.sidebar.slider(
    "Expected anomaly fraction (contamination)", 0.01, 0.20, 0.05, 0.01
)
n_components = st.sidebar.slider("PCA components (for modeling)", 2, min(8, len(feature_cols)), 3)

if len(df) < 50 or len(X_raw) < 50:
    st.error("Filters removed too much data — please relax filters.")
    st.stop()

# ======================================================================
# SCALING + PCA (dimension reduction)
# ======================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

pca_full = PCA(n_components=min(len(feature_cols), 8), random_state=42)
pca_full.fit(X_scaled)
explained_var = pca_full.explained_variance_ratio_

pca_model = PCA(n_components=n_components, random_state=42)
X_pca = pca_model.fit_transform(X_scaled)
pca_cols = [f"PC{i+1}" for i in range(n_components)]
pca_df = pd.DataFrame(X_pca, columns=pca_cols)

# ======================================================================
# ANOMALY DETECTION — 4+ ALGORITHMS
# ======================================================================
@st.cache_data(show_spinner=False)
def run_models(X_scaled_arr, contamination, seed=42):
    n = X_scaled_arr.shape[0]
    results = {}

    # 1. Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
    iso_pred = iso.fit_predict(X_scaled_arr)  # -1 anomaly, 1 normal
    iso_score = -iso.score_samples(X_scaled_arr)  # higher = more anomalous
    results["Isolation Forest"] = {"pred": iso_pred, "score": iso_score}

    # 2. Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    lof_pred = lof.fit_predict(X_scaled_arr)
    lof_score = -lof.negative_outlier_factor_
    results["Local Outlier Factor"] = {"pred": lof_pred, "score": lof_score}

    # 3. One-Class SVM
    ocsvm = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
    ocsvm_pred = ocsvm.fit_predict(X_scaled_arr)
    ocsvm_score = -ocsvm.decision_function(X_scaled_arr)
    results["One-Class SVM"] = {"pred": ocsvm_pred, "score": ocsvm_score}

    # 4. Elliptic Envelope (robust covariance / Mahalanobis distance)
    try:
        ee = EllipticEnvelope(contamination=contamination, random_state=seed)
        ee_pred = ee.fit_predict(X_scaled_arr)
        ee_score = -ee.score_samples(X_scaled_arr)
        results["Elliptic Envelope"] = {"pred": ee_pred, "score": ee_score}
    except Exception:
        pass

    # 5. DBSCAN (density based; noise points = -1 treated as anomalies)
    db = DBSCAN(eps=1.2, min_samples=10)
    db_labels = db.fit_predict(X_scaled_arr)
    db_pred = np.where(db_labels == -1, -1, 1)
    # score = distance-ish proxy: 1 if noise else 0, refine with cluster size
    db_score = (db_labels == -1).astype(float)
    results["DBSCAN"] = {"pred": db_pred, "score": db_score}

    return results


with st.spinner("Running anomaly detection models..."):
    model_results = run_models(X_scaled, contamination)

algo_names = list(model_results.keys())
for name, r in model_results.items():
    df[f"flag_{name}"] = (r["pred"] == -1)
    # normalize score 0-1 for comparability
    s = r["score"]
    s_norm = (s - s.min()) / (s.max() - s.min() + 1e-9)
    df[f"score_{name}"] = s_norm

df["consensus_votes"] = df[[f"flag_{n}" for n in algo_names]].sum(axis=1)
df["consensus_anomaly"] = df["consensus_votes"] >= 2  # flagged by 2+ methods

# ======================================================================
# TABS
# ======================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["\U0001F4CA Explore & Visualize", "\U0001F53B Dimension Reduction",
     "\U0001F6A8 Anomaly Detection", "\u2705 Validation", "\U0001F50D Flight Drill-Down"]
)

# ----------------------------------------------------------------------
# TAB 1 — EDA
# ----------------------------------------------------------------------
with tab1:
    st.header("Explore & Visualize the Dataset")
    if use_synthetic:
        st.warning(
            "\U0001F9EA Running on **synthetic demo data**, not your real RITA/BTS file. "
            "Uncheck 'Use synthetic demo data' in the sidebar once your CSV is ready."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flights (analyzed)", f"{len(df):,}")
    if "dep_delay" in colmap:
        c2.metric("Avg Dep Delay (min)", f"{df[colmap['dep_delay']].mean():.1f}")
    if "arr_delay" in colmap:
        c3.metric("Avg Arr Delay (min)", f"{df[colmap['arr_delay']].mean():.1f}")
    if "carrier" in colmap:
        c4.metric("Carriers", df[colmap["carrier"]].nunique())

    col1, col2 = st.columns(2)
    with col1:
        if "dep_delay" in colmap:
            fig = px.histogram(df, x=colmap["dep_delay"], nbins=80,
                                title="Departure Delay Distribution")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "arr_delay" in colmap:
            fig = px.histogram(df, x=colmap["arr_delay"], nbins=80,
                                title="Arrival Delay Distribution", color_discrete_sequence=["orange"])
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if "carrier" in colmap and "arr_delay" in colmap:
            g = df.groupby(colmap["carrier"])[colmap["arr_delay"]].mean().sort_values()
            fig = px.bar(g, title="Avg Arrival Delay by Carrier", labels={"value": "Avg Delay (min)"})
            st.plotly_chart(fig, use_container_width=True)
    with col4:
        if "origin" in colmap and "dest" in colmap:
            df["ROUTE"] = df[colmap["origin"]] + " \u2192 " + df[colmap["dest"]]
            top_routes = df["ROUTE"].value_counts().head(15)
            fig = px.bar(top_routes, orientation="h", title="Top 15 Busiest Routes")
            st.plotly_chart(fig, use_container_width=True)

    if "crs_dep" in colmap:
        df["DEP_HOUR"] = (df[colmap["crs_dep"]] // 100).clip(0, 23)
        col5, col6 = st.columns(2)
        with col5:
            if "arr_delay" in colmap:
                g2 = df.groupby("DEP_HOUR")[colmap["arr_delay"]].mean()
                fig = px.line(g2, markers=True, title="Avg Arrival Delay by Hour of Day")
                st.plotly_chart(fig, use_container_width=True)
        with col6:
            fig = px.histogram(df, x="DEP_HOUR", nbins=24, title="Flight Volume by Hour of Day")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap (numeric features)")
    corr = X_raw.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title="Feature Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Boxplots — Outlier Spotting per Feature")
    feat_for_box = st.selectbox("Select feature", feature_cols, key="box_feat")
    fig = px.box(X_raw, y=feat_for_box, points="outliers", title=f"Boxplot of {feat_for_box}")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# TAB 2 — DIMENSION REDUCTION
# ----------------------------------------------------------------------
with tab2:
    st.header("Dimension Reduction (PCA)")
    st.markdown(
        f"Original feature space has **{len(feature_cols)}** numeric dimensions: "
        f"`{', '.join(feature_cols)}`. PCA is used to compress this into "
        f"**{n_components}** components while retaining as much variance as possible, "
        "both for visualization and to speed up / stabilize the anomaly detection models."
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            x=[f"PC{i+1}" for i in range(len(explained_var))],
            y=explained_var,
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

    st.markdown(
        f"Using **{n_components}** components explains "
        f"**{cum_var[n_components-1]*100:.1f}%** of total variance."
    )

    st.subheader("Flights Projected onto Principal Components")
    color_by = colmap.get("arr_delay") if "arr_delay" in colmap else None
    plot_df = pca_df.copy()
    if color_by:
        plot_df["color_val"] = df[color_by].values
    if n_components >= 3:
        fig = px.scatter_3d(
            plot_df, x="PC1", y="PC2", z="PC3",
            color="color_val" if color_by else None,
            color_continuous_scale="RdYlGn_r",
            title="3D PCA Projection (colored by arrival delay)",
            opacity=0.6,
        )
    else:
        fig = px.scatter(
            plot_df, x="PC1", y="PC2",
            color="color_val" if color_by else None,
            color_continuous_scale="RdYlGn_r",
            title="2D PCA Projection (colored by arrival delay)",
            opacity=0.6,
        )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("PCA Loadings (feature contribution to each component)")
    loadings = pd.DataFrame(
        pca_model.components_.T, index=feature_cols, columns=pca_cols
    )
    fig = px.imshow(loadings, text_auto=".2f", color_continuous_scale="RdBu_r",
                     title="Loadings: how each original feature maps into PCA space")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# TAB 3 — ANOMALY DETECTION
# ----------------------------------------------------------------------
with tab3:
    st.header("Anomaly Detection — 4+ Algorithms")
    st.markdown(
        "Each algorithm below is applied to the standardized feature space "
        f"(contamination / expected anomaly rate = **{contamination:.0%}**). "
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
    fig = px.scatter(
        plot_df2, x="PC1", y="PC2", color="Anomaly",
        color_discrete_map={"Anomaly": "red", "Normal": "lightblue"},
        hover_data=["Score"],
        title=f"{algo_pick}: Anomalies in PCA Space",
        opacity=0.7,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Consensus View — Voting Across All Algorithms")
    st.markdown(
        "Each flight gets one vote per algorithm that flags it. "
        "Flights flagged by **2 or more** algorithms are treated as high-confidence anomalies."
    )
    vote_counts = df["consensus_votes"].value_counts().sort_index()
    fig = px.bar(vote_counts, title="Number of Flights by Vote Count",
                 labels={"index": "Number of algorithms flagging flight", "value": "Number of flights"})
    st.plotly_chart(fig, use_container_width=True)

    plot_df3 = pca_df.copy()
    plot_df3["Votes"] = df["consensus_votes"]
    fig = px.scatter(
        plot_df3, x="PC1", y="PC2", color="Votes",
        color_continuous_scale="Reds",
        title="Consensus Anomaly Score in PCA Space (darker = more algorithms agree)",
        opacity=0.7,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Consensus Anomalies")
    show_cols = [c for c in [colmap.get("date"), colmap.get("carrier"), colmap.get("flight_num"),
                              colmap.get("origin"), colmap.get("dest"), colmap.get("dep_delay"),
                              colmap.get("arr_delay")] if c] + ["consensus_votes"]
    top_anom = df.sort_values("consensus_votes", ascending=False)[show_cols].head(20)
    st.dataframe(top_anom, use_container_width=True)

# ----------------------------------------------------------------------
# TAB 4 — VALIDATION
# ----------------------------------------------------------------------
with tab4:
    st.header("Validating the Anomaly Detection Results")
    st.markdown(
        "No single algorithm is ground truth, so we validate by checking **agreement** "
        "between methods. Consistent overlap across very different algorithms "
        "(density-based, tree-based, distance-based, covariance-based) is stronger "
        "evidence that flagged flights are genuinely anomalous rather than an artifact "
        "of one model's assumptions."
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

    st.subheader("Sanity Check — Do flagged flights actually look extreme?")
    st.markdown(
        "If detection is working, consensus anomalies should show noticeably higher "
        "delays / more extreme values than normal flights on average."
    )
    compare_cols = [c for c in [colmap.get("dep_delay"), colmap.get("arr_delay"),
                                 colmap.get("taxi_out"), "ELAPSED_DIFF"] if c and c in df.columns]
    comp = df.groupby("consensus_anomaly")[compare_cols].mean().rename(
        index={True: "Consensus Anomaly", False: "Normal"}
    )
    st.dataframe(comp.style.format("{:.1f}"), use_container_width=True)

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
            st.caption("Closer to +1 means the two groups (normal / anomaly) are well separated in feature space.")
    except Exception as e:
        st.caption(f"Silhouette score unavailable: {e}")

# ----------------------------------------------------------------------
# TAB 5 — FLIGHT DRILL-DOWN
# ----------------------------------------------------------------------
with tab5:
    st.header("Flight Drill-Down: Why Was This Flight Flagged?")

    ranked = df.sort_values("consensus_votes", ascending=False).reset_index()
    ranked = ranked.rename(columns={"index": "row_id"})

    label_cols = [c for c in [colmap.get("date"), colmap.get("carrier"),
                               colmap.get("flight_num"), colmap.get("origin"),
                               colmap.get("dest")] if c]

    def make_label(row):
        parts = [str(row[c]) for c in label_cols]
        return " | ".join(parts) + f"  (votes={row['consensus_votes']})"

    ranked["label"] = ranked.apply(make_label, axis=1)

    st.markdown("Pick a flight below (defaults to the most suspicious one) to inspect it in detail.")
    choice_label = st.selectbox("Select a flight", ranked["label"].tolist())
    sel_row = ranked[ranked["label"] == choice_label].iloc[0]
    row_id = int(sel_row["row_id"])

    flight_row = df.loc[row_id]
    x_row = X_raw.loc[row_id]

    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithms flagging this flight", f"{int(flight_row['consensus_votes'])} / {len(algo_names)}")
    if "dep_delay" in colmap:
        c2.metric("Departure Delay", f"{flight_row[colmap['dep_delay']]:.0f} min")
    if "arr_delay" in colmap:
        c3.metric("Arrival Delay", f"{flight_row[colmap['arr_delay']]:.0f} min")

    st.subheader("Raw Flight Record")
    show_fields = [c for c in [colmap.get("date"), colmap.get("carrier"), colmap.get("flight_num"),
                                colmap.get("origin"), colmap.get("dest"), colmap.get("dep_delay"),
                                colmap.get("arr_delay"), colmap.get("taxi_out"), colmap.get("taxi_in"),
                                colmap.get("air_time"), colmap.get("distance")] if c]
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
        explain_df.head(8), x="Z-score (std devs from average)", y="Feature",
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
        f"It was flagged by **{int(flight_row['consensus_votes'])} out of {len(algo_names)}** "
        f"anomaly detection algorithms."
    )

    st.subheader("Where this flight sits relative to everyone else (PCA space)")
    highlight_df = pca_df.copy()
    highlight_df["Type"] = "Other flights"
    highlight_df.loc[row_id, "Type"] = "Selected flight"
    fig = px.scatter(
        highlight_df, x="PC1", y="PC2", color="Type",
        color_discrete_map={"Other flights": "lightgray", "Selected flight": "red"},
        title="Selected Flight Highlighted in PCA Space",
    )
    fig.update_traces(marker=dict(size=10), selector=dict(name="Selected flight"))
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Flight Anomaly Control Tower — RITA/BTS On-Time Performance Data")