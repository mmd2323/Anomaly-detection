"""
Flight Anomaly Control Tower — Route-Level Edition
=====================================
Interactive Streamlit dashboard for detecting anomalous FLIGHT ROUTES
using an ALREADY-TRANSFORMED, route-level dataset (one row per route,
aggregated across all flights on that route).

Observational unit = ROUTE, not individual flight -- this is the required
change of observational unit: individual flight delays are mostly noise
(weather, ATC, a late inbound aircraft); a route behaving unusually
compared to its peers is a real, explainable, statistically meaningful
anomaly.

Expected columns in the CSV:
    flight_path, number_of_flights, mean_departure_delay,
    probablility_of_dep_delay, max_departure_delay, mean_arrival_delay,
    probablility_of_arr_delay, max_arrival_delay

Run with:
    streamlit run dash_flightpath.py

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

# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="Flight Anomaly Control Tower — Route Level",
    layout="wide",
    page_icon="\u2708\ufe0f",
)

# ======================================================================
# DATA LOADING — the CSV is ALREADY route-level, no aggregation needed here
# ======================================================================
# Update this path if your file lives somewhere else, then re-run.
DATA_FILE = r"C:\Users\jaina\.spyder-py3\Flights.csv"

ROUTE_FEATURE_COLS = [
    "number_of_flights", "mean_departure_delay", "probablility_of_dep_delay",
    "max_departure_delay", "mean_arrival_delay", "probablility_of_arr_delay",
    "max_arrival_delay",
]


@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file, low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = df.columns.str.strip()  # defensive: strip stray whitespace (e.g. from Excel re-saves)
    return df


# ======================================================================
# GENERIC PCA HELPER -- works on ANY numeric dataframe, any column set.
# Never assumes specific column names or a specific number of features,
# and always hands back a PCA-space dataframe with a clean RangeIndex so
# it can be safely joined back to the source dataframe positionally.
# ======================================================================
def run_pca(numeric_df: pd.DataFrame, n_components: int):
    numeric_df = numeric_df.reset_index(drop=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df.values)

    max_components = min(numeric_df.shape[1], 8)
    pca_full = PCA(n_components=max_components, random_state=42)
    pca_full.fit(X_scaled)

    n_components = max(2, min(n_components, numeric_df.shape[1]))
    pca_model = PCA(n_components=n_components, random_state=42)
    X_pca = pca_model.fit_transform(X_scaled)
    pca_cols = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=pca_cols)  # default RangeIndex -- matches numeric_df positionally

    return {
        "scaler": scaler, "X_scaled": X_scaled,
        "pca_full": pca_full, "explained_var": pca_full.explained_variance_ratio_,
        "pca_model": pca_model, "pca_df": pca_df, "pca_cols": pca_cols,
        "feature_names": list(numeric_df.columns),
    }


# ======================================================================
# GENERIC ANOMALY DETECTION HELPER -- 5 algorithms, any numeric array
# ======================================================================
MAX_TRAIN_N_FOR_QUADRATIC_MODELS = 20000  # LOF / One-Class SVM are O(n^2); a safety net, not a bottleneck at route-level scale


@st.cache_data(show_spinner=False)
def run_models(X_scaled_arr, contamination, seed=42, max_train_n=MAX_TRAIN_N_FOR_QUADRATIC_MODELS):
    results = {}
    n = X_scaled_arr.shape[0]

    iso = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
    iso_pred = iso.fit_predict(X_scaled_arr)
    iso_score = -iso.score_samples(X_scaled_arr)
    results["Isolation Forest"] = {"pred": iso_pred, "score": iso_score}

    if n > max_train_n:
        rng = np.random.default_rng(seed)
        train_idx = rng.choice(n, size=max_train_n, replace=False)
        X_train = X_scaled_arr[train_idx]
    else:
        X_train = X_scaled_arr

    lof = LocalOutlierFactor(n_neighbors=min(20, max(2, n - 1)), contamination=contamination, novelty=True)
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

    db = DBSCAN(eps=1.2, min_samples=5)
    db_labels = db.fit_predict(X_scaled_arr)
    db_pred = np.where(db_labels == -1, -1, 1)
    db_score = (db_labels == -1).astype(float)
    results["DBSCAN"] = {"pred": db_pred, "score": db_score}

    return results


# ======================================================================
# LOAD DATA
# ======================================================================
st.sidebar.title("\u2708\ufe0f Control Tower Settings")

try:
    raw_df = load_data(DATA_FILE)
except Exception as e:
    st.title("\u2708\ufe0f Flight Anomaly Control Tower")
    st.error(f"Could not load '{DATA_FILE}': {e}\n\nUpdate DATA_FILE near the top of the script.")
    st.stop()

missing = set(ROUTE_FEATURE_COLS + ["flight_path"]) - set(raw_df.columns)
if missing:
    st.error(f"This file is missing expected columns: {missing}. Columns found: {raw_df.columns.tolist()}")
    st.stop()

st.sidebar.success(f"Loaded: {len(raw_df):,} routes")

# ======================================================================
# SIDEBAR — only the controls that are actually still relevant at this
# dataset size (~5,000 routes, 7 numeric features). Removed: day-of-week
# filter, destination-state filter, and "rows to aggregate" slider --
# none of those columns exist in this already-aggregated file.
# ======================================================================
with st.sidebar.form("controls_form"):
    st.subheader("Settings")
    min_flights_per_route = st.slider(
        "Minimum flights per route (excludes one-off routes)",
        1, int(raw_df["number_of_flights"].quantile(0.95)), 5,
        help="A route with only 1-2 flights all month can't reliably be called 'anomalous'.",
    )
    contamination = st.slider("Expected anomaly fraction (contamination)", 0.01, 0.20, 0.05, 0.01)
    n_components = st.slider("PCA components", 2, len(ROUTE_FEATURE_COLS), 3)
    submitted = st.form_submit_button("\u25B6 Apply & Run Analysis", use_container_width=True)

st.sidebar.caption("Nothing recomputes until you click 'Apply & Run Analysis' above.")

flight_path_df = raw_df[raw_df["number_of_flights"] >= min_flights_per_route].reset_index(drop=True)

if len(flight_path_df) < 20:
    st.error(
        f"Only {len(flight_path_df)} routes remain after filtering — too few for reliable PCA / anomaly "
        f"detection. Lower 'Minimum flights per route'."
    )
    st.stop()

# ------------------------------------------------------------------
# PCA on the ROUTE-LEVEL feature set
# ------------------------------------------------------------------
route_numeric = flight_path_df[ROUTE_FEATURE_COLS]
pca_out = run_pca(route_numeric, n_components)
X_scaled = pca_out["X_scaled"]
pca_full = pca_out["pca_full"]
explained_var = pca_out["explained_var"]
pca_model = pca_out["pca_model"]
pca_df = pca_out["pca_df"]
pca_cols = pca_out["pca_cols"]

# ------------------------------------------------------------------
# Anomaly detection on the ROUTE-LEVEL feature set
# ------------------------------------------------------------------
with st.spinner("Running anomaly detection models on route-level data..."):
    model_results = run_models(X_scaled, contamination)

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
     "\U0001F6A8 Anomaly Detection", "\u2705 Validation", "\U0001F50D Route Drill-Down"]
)

# ----------------------------------------------------------------------
# TAB 1 — EDA, entirely at the route level (no flight-level data loaded)
# ----------------------------------------------------------------------
with tab1:
    st.header("Explore & Visualize the Dataset")
    st.info(
        f"**Observational unit = ROUTE.** This dataset has **{len(flight_path_df):,} routes** "
        f"(after excluding routes with fewer than {min_flights_per_route} flights), each described "
        f"by {len(ROUTE_FEATURE_COLS)} aggregate features."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Routes analyzed", f"{len(flight_path_df):,}")
    c2.metric("Avg flights per route", f"{flight_path_df['number_of_flights'].mean():.0f}")
    c3.metric("Avg mean arrival delay", f"{flight_path_df['mean_arrival_delay'].mean():.1f} min")
    c4.metric("Avg P(arrival delayed)", f"{flight_path_df['probablility_of_arr_delay'].mean()*100:.1f}%")

    st.subheader("Feature Distributions")
    feat_pick = st.selectbox("Feature", ROUTE_FEATURE_COLS, key="route_feat_hist")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(flight_path_df, x=feat_pick, nbins=60, title=f"Distribution of {feat_pick} across routes")
        fig.add_vline(x=flight_path_df[feat_pick].mean(), line_dash="dash", line_color="red", annotation_text="mean")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(flight_path_df, y=feat_pick, points="outliers", title=f"Boxplot of {feat_pick} across routes")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Busiest Routes")
    top_volume = flight_path_df.sort_values("number_of_flights", ascending=False).head(20)
    fig = px.bar(top_volume, x="number_of_flights", y="flight_path", orientation="h",
                 title="Top 20 Routes by Flight Volume", labels={"flight_path": "Route", "number_of_flights": "Flights"})
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Worst Routes by Average Arrival Delay")
    worst_delay = flight_path_df.sort_values("mean_arrival_delay", ascending=False).head(20)
    fig = px.bar(worst_delay, x="mean_arrival_delay", y="flight_path", orientation="h",
                 title="Top 20 Routes by Mean Arrival Delay", labels={"flight_path": "Route", "mean_arrival_delay": "Avg Arr Delay (min)"},
                 color="mean_arrival_delay", color_continuous_scale="Reds")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("3D Relationship Between Route-Level Features")
    fig = px.scatter_3d(
        flight_path_df, x="mean_departure_delay", y="mean_arrival_delay", z="number_of_flights",
        color="max_arrival_delay", color_continuous_scale="RdYlGn_r", opacity=0.6,
        title="Mean Dep Delay vs Mean Arr Delay vs Flight Volume, colored by worst single delay",
        hover_data=["flight_path"],
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Departure vs Arrival Delay Probability")
    fig = px.scatter(
        flight_path_df, x="probablility_of_dep_delay", y="probablility_of_arr_delay",
        size="number_of_flights", color="mean_arrival_delay", color_continuous_scale="RdYlGn_r",
        opacity=0.6, hover_data=["flight_path"],
        title="P(Departure Delayed) vs P(Arrival Delayed), sized by flight volume",
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="gray", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pairwise Feature Relationships (Scatter Matrix)")
    splom_df = route_numeric.copy()
    fig = px.scatter_matrix(splom_df, dimensions=ROUTE_FEATURE_COLS, opacity=0.35,
                             title="Scatter Matrix of All Route-Level Features")
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = route_numeric.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title="Feature Correlation Matrix (route-level)")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# TAB 2 — DIMENSION REDUCTION (route-level, generic PCA helper)
# ----------------------------------------------------------------------
with tab2:
    st.header("Dimension Reduction (PCA) — Route Level")
    st.markdown(
        f"Route-level feature space has **{len(ROUTE_FEATURE_COLS)}** numeric dimensions: "
        f"`{', '.join(ROUTE_FEATURE_COLS)}`. PCA compresses this into **{len(pca_cols)}** components. "
        f"This uses a generic `run_pca()` helper that works on any numeric dataframe — swap in a "
        f"completely different feature set and this section keeps working unchanged."
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(x=[f"PC{i+1}" for i in range(len(explained_var))], y=explained_var,
                     title="Explained Variance per Principal Component",
                     labels={"x": "Component", "y": "Explained Variance Ratio"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        cum_var = np.cumsum(explained_var)
        fig = px.line(x=[f"PC{i+1}" for i in range(len(cum_var))], y=cum_var, markers=True,
                      title="Cumulative Explained Variance")
        fig.add_hline(y=0.9, line_dash="dash", line_color="red", annotation_text="90% threshold")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"Using **{len(pca_cols)}** components explains **{cum_var[len(pca_cols)-1]*100:.1f}%** of total variance.")

    st.subheader("Routes Projected onto Principal Components")
    plot_df = pca_df.copy()
    plot_df["color_val"] = flight_path_df["mean_arrival_delay"].values
    plot_df["flight_path"] = flight_path_df["flight_path"].values
    if len(pca_cols) >= 3:
        fig = px.scatter_3d(plot_df, x="PC1", y="PC2", z="PC3", color="color_val",
                             color_continuous_scale="RdYlGn_r", hover_data=["flight_path"],
                             title="3D PCA Projection (colored by mean arrival delay)", opacity=0.6)
    else:
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="color_val",
                          color_continuous_scale="RdYlGn_r", hover_data=["flight_path"],
                          title="2D PCA Projection (colored by mean arrival delay)", opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("2D PCA Biplot (PC1 vs PC2) with Loading Vectors")
    st.caption("Arrows show how much each original feature contributes to PC1/PC2 and in which direction.")
    fig = px.scatter(plot_df, x="PC1", y="PC2", opacity=0.35, title="PCA Biplot: Routes (points) + Feature Loadings (arrows)")
    scale = 3 * plot_df[["PC1", "PC2"]].abs().quantile(0.95).max()
    loadings_2d = pd.DataFrame(pca_model.components_[:2].T, index=ROUTE_FEATURE_COLS, columns=["PC1", "PC2"])
    for feat, row in loadings_2d.iterrows():
        fig.add_shape(type="line", x0=0, y0=0, x1=row["PC1"] * scale, y1=row["PC2"] * scale,
                       line=dict(color="black", width=2))
        fig.add_annotation(x=row["PC1"] * scale, y=row["PC2"] * scale, text=feat,
                            showarrow=False, font=dict(color="black", size=11))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("PCA Loadings (feature contribution to each component)")
    loadings = pd.DataFrame(pca_model.components_.T, index=ROUTE_FEATURE_COLS, columns=pca_cols)
    fig = px.imshow(loadings, text_auto=".2f", color_continuous_scale="RdBu_r",
                     title="Loadings: how each original route-level feature maps into PCA space")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# TAB 3 — ANOMALY DETECTION (route-level)
# ----------------------------------------------------------------------
with tab3:
    st.header("Anomaly Detection — 5 Algorithms (Route Level)")
    st.markdown(
        f"Contamination / expected anomaly rate = **{contamination:.0%}** of **{len(flight_path_df):,} routes**. "
        "A route is flagged (\U0001F6A8) if the model considers its aggregate behavior an outlier."
    )

    counts = {name: int(flight_path_df[f"flag_{name}"].sum()) for name in algo_names}
    cols = st.columns(len(algo_names))
    for c, name in zip(cols, algo_names):
        c.metric(name, f"{counts[name]:,} flagged", f"{counts[name]/len(flight_path_df)*100:.1f}%")

    st.subheader("Where the anomalies sit in PCA space")
    algo_pick = st.selectbox("Choose algorithm to visualize", algo_names)
    plot_df2 = pca_df.copy()
    plot_df2["Anomaly"] = flight_path_df[f"flag_{algo_pick}"].map({True: "Anomaly", False: "Normal"}).values
    plot_df2["Score"] = flight_path_df[f"score_{algo_pick}"].values
    plot_df2["flight_path"] = flight_path_df["flight_path"].values
    if len(pca_cols) >= 3:
        fig = px.scatter_3d(plot_df2, x="PC1", y="PC2", z="PC3", color="Anomaly",
                             color_discrete_map={"Anomaly": "red", "Normal": "lightblue"},
                             hover_data=["Score", "flight_path"], title=f"{algo_pick}: Anomalies in PCA Space", opacity=0.6)
    else:
        fig = px.scatter(plot_df2, x="PC1", y="PC2", color="Anomaly",
                          color_discrete_map={"Anomaly": "red", "Normal": "lightblue"},
                          hover_data=["Score", "flight_path"], title=f"{algo_pick}: Anomalies in PCA Space", opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Anomaly Score Distribution")
    score_series = flight_path_df[f"score_{algo_pick}"]
    thresh = score_series[flight_path_df[f"flag_{algo_pick}"]].min() if flight_path_df[f"flag_{algo_pick}"].any() else score_series.max()
    fig = px.histogram(flight_path_df, x=f"score_{algo_pick}", nbins=60, title=f"{algo_pick}: Score Distribution")
    fig.add_vline(x=thresh, line_dash="dash", line_color="red", annotation_text="flag threshold")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Consensus View — Voting Across All Algorithms")
    st.markdown("A route is a **high-confidence anomaly** if 2+ algorithms independently flag it.")
    vote_counts = flight_path_df["consensus_votes"].value_counts().sort_index()
    fig = px.bar(vote_counts, title="Number of Routes by Vote Count",
                 labels={"index": "Number of algorithms flagging route", "value": "Number of routes"})
    st.plotly_chart(fig, use_container_width=True)

    plot_df3 = pca_df.copy()
    plot_df3["Votes"] = flight_path_df["consensus_votes"].values
    plot_df3["flight_path"] = flight_path_df["flight_path"].values
    if len(pca_cols) >= 3:
        fig = px.scatter_3d(plot_df3, x="PC1", y="PC2", z="PC3", color="Votes",
                             color_continuous_scale="Reds", hover_data=["flight_path"],
                             title="Consensus Anomaly Score in PCA Space", opacity=0.6)
    else:
        fig = px.scatter(plot_df3, x="PC1", y="PC2", color="Votes", color_continuous_scale="Reds",
                          hover_data=["flight_path"], title="Consensus Anomaly Score in PCA Space", opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Consensus Anomalies")
    show_cols = ["flight_path"] + ROUTE_FEATURE_COLS + ["consensus_votes"]
    top_anom = flight_path_df.sort_values("consensus_votes", ascending=False)[show_cols].head(20)
    st.dataframe(top_anom, use_container_width=True)

# ----------------------------------------------------------------------
# TAB 4 — VALIDATION (route-level)
# ----------------------------------------------------------------------
with tab4:
    st.header("Validating the Anomaly Detection Results")
    st.markdown(
        "No single algorithm is ground truth, so we validate via agreement between methods. "
        "Consistent overlap across very different algorithms (density-based, tree-based, "
        "distance-based, covariance-based) is stronger evidence flagged routes are genuinely "
        "anomalous, not an artifact of one model's assumptions."
    )

    st.subheader("Pairwise Agreement (Jaccard Similarity of Flagged Sets)")
    jaccard = pd.DataFrame(index=algo_names, columns=algo_names, dtype=float)
    for a in algo_names:
        for b in algo_names:
            set_a = set(flight_path_df.index[flight_path_df[f"flag_{a}"]])
            set_b = set(flight_path_df.index[flight_path_df[f"flag_{b}"]])
            union = len(set_a | set_b)
            inter = len(set_a & set_b)
            jaccard.loc[a, b] = inter / union if union > 0 else 0.0
    fig = px.imshow(jaccard.astype(float), text_auto=".2f", color_continuous_scale="Viridis",
                     title="Jaccard Similarity Between Algorithms' Flagged Routes")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rank Correlation of Anomaly Scores (Spearman)")
    score_cols = [f"score_{n}" for n in algo_names]
    spearman_corr = flight_path_df[score_cols].corr(method="spearman")
    spearman_corr.columns = algo_names
    spearman_corr.index = algo_names
    fig = px.imshow(spearman_corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title="Spearman Correlation Between Algorithm Anomaly Scores")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Direct Score Comparison Between Two Algorithms")
    col_x, col_y = st.columns(2)
    with col_x:
        algo_x = st.selectbox("Algorithm A (x-axis)", algo_names, index=0, key="valx")
    with col_y:
        algo_y = st.selectbox("Algorithm B (y-axis)", algo_names, index=min(1, len(algo_names) - 1), key="valy")
    comp_scores_df = pd.DataFrame({
        algo_x: flight_path_df[f"score_{algo_x}"], algo_y: flight_path_df[f"score_{algo_y}"],
        "Consensus": flight_path_df["consensus_anomaly"].map({True: "Consensus Anomaly", False: "Normal"}),
    })
    fig = px.scatter(comp_scores_df, x=algo_x, y=algo_y, color="Consensus",
                      color_discrete_map={"Consensus Anomaly": "red", "Normal": "lightblue"},
                      opacity=0.5, title=f"{algo_x} Score vs {algo_y} Score")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sanity Check — Do flagged routes actually look extreme?")
    comp = flight_path_df.groupby("consensus_anomaly")[ROUTE_FEATURE_COLS].mean().rename(
        index={True: "Consensus Anomaly", False: "Normal"}
    )
    st.dataframe(comp.style.format("{:.2f}"), use_container_width=True)

    fig = go.Figure()
    for c in ["mean_arrival_delay", "max_arrival_delay", "probablility_of_arr_delay"]:
        fig.add_trace(go.Bar(name=c, x=["Normal", "Consensus Anomaly"],
                              y=[comp.loc["Normal", c], comp.loc["Consensus Anomaly", c]]))
    fig.update_layout(barmode="group", title="Average Feature Values: Normal vs Consensus Anomaly Routes")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Silhouette-style separation check (optional statistic)")
    try:
        from sklearn.metrics import silhouette_score
        labels_bin = flight_path_df["consensus_anomaly"].astype(int).values
        if 0 < labels_bin.sum() < len(labels_bin):
            sil = silhouette_score(X_scaled, labels_bin)
            st.metric("Silhouette score (normal vs consensus anomaly)", f"{sil:.3f}")
            st.caption("Closer to +1 means the two groups are well separated in feature space.")
    except Exception as e:
        st.caption(f"Silhouette score unavailable: {e}")

# ----------------------------------------------------------------------
# TAB 5 — ROUTE DRILL-DOWN
# ----------------------------------------------------------------------
with tab5:
    st.header("Route Drill-Down: Why Was This Route Flagged?")

    ranked = flight_path_df.sort_values("consensus_votes", ascending=False).reset_index(drop=True)
    ranked["label"] = ranked.apply(
        lambda r: f"{r['flight_path']} (votes={r['consensus_votes']}, n={int(r['number_of_flights'])})", axis=1
    )

    st.markdown("Pick a route below (defaults to the most suspicious one) to inspect it in detail.")
    choice_label = st.selectbox("Select a route", ranked["label"].tolist())
    label = choice_label.split(" (")[0]
    row_id = int(flight_path_df.index[flight_path_df["flight_path"] == label].tolist()[0])
    route_row = flight_path_df.iloc[row_id]


    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithms flagging this route", f"{int(route_row['consensus_votes'])} / {len(algo_names)}")
    c2.metric("Number of Flights", f"{int(route_row['number_of_flights'])}")
    c3.metric("Mean Arrival Delay", f"{route_row['mean_arrival_delay']:.1f} min")

    st.subheader("Route Summary")
    st.dataframe(route_row[["flight_path"] + ROUTE_FEATURE_COLS].to_frame().T, use_container_width=True)

    st.subheader("Per-Algorithm Verdict")
    verdict_df = pd.DataFrame({
        "Algorithm": algo_names,
        "Flagged?": [bool(route_row[f"flag_{n}"]) for n in algo_names],
        "Anomaly Score (0-1, normalized)": [round(float(route_row[f"score_{n}"]), 3) for n in algo_names],
    })
    st.dataframe(verdict_df, use_container_width=True)

    st.subheader("Why is this route anomalous? (Feature Deviation Explanation)")
    means = route_numeric.mean()
    stds = route_numeric.std().replace(0, 1)
    x_row = route_numeric.iloc[row_id]
    z = (x_row - means) / stds
    z_sorted = z.reindex(z.abs().sort_values(ascending=False).index)

    explain_df = pd.DataFrame({
        "Feature": z_sorted.index,
        "This route's value": [x_row[f] for f in z_sorted.index],
        "Average across all routes": [means[f] for f in z_sorted.index],
        "Z-score (std devs from average)": z_sorted.values,
    })
    st.dataframe(explain_df.style.format({
        "This route's value": "{:.2f}", "Average across all routes": "{:.2f}",
        "Z-score (std devs from average)": "{:.2f}",
    }), use_container_width=True)

    fig = px.bar(explain_df, x="Z-score (std devs from average)", y="Feature", orientation="h",
                 color="Z-score (std devs from average)", color_continuous_scale="RdBu_r",
                 title="Top Contributing Features (largest deviation from a typical route)")
    st.plotly_chart(fig, use_container_width=True)

    top_feat = z_sorted.index[0]
    top_z = z_sorted.iloc[0]
    direction = "higher" if top_z > 0 else "lower"
    st.info(
        f"**Plain-language summary:** the route **{route_row['flight_path']}** is most unusual because its "
        f"**{top_feat}** ({x_row[top_feat]:.2f}) is **{abs(top_z):.1f} standard deviations {direction}** "
        f"than the typical route (average = {means[top_feat]:.2f}). It was flagged by "
        f"**{int(route_row['consensus_votes'])} out of {len(algo_names)}** algorithms."
    )

    st.subheader("Where this route sits relative to all others (PCA space)")
    highlight_df = pca_df.copy()
    highlight_df["Type"] = "Other routes"
    highlight_df.loc[row_id, "Type"] = "Selected route"
    if len(pca_cols) >= 3:
        fig = px.scatter_3d(highlight_df, x="PC1", y="PC2", z="PC3", color="Type",
                             color_discrete_map={"Other routes": "lightgray", "Selected route": "red"},
                             title="Selected Route Highlighted in PCA Space")
    else:
        fig = px.scatter(highlight_df, x="PC1", y="PC2", color="Type",
                          color_discrete_map={"Other routes": "lightgray", "Selected route": "red"},
                          title="Selected Route Highlighted in PCA Space")
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Flight Anomaly Control Tower — Route-Level Analysis")