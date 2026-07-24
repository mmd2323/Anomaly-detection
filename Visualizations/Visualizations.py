"""
Flight Anomaly Control Tower — static figure generator
======================================================
Reproduces every figure used in the project report, straight from the raw
flight-level extract. Unlike the Streamlit dashboard (which renders Plotly
charts live in the browser), this script writes PNG files to disk so they
can be embedded in the report and version-controlled.

Run from the repo root:
    python Visualizations/generate_visualizations.py

Outputs (written to Visualizations/figures/):
    fig1_delay_distribution.png   fig4_top_anomalies.png
    fig2_pca_anomalies.png        fig5_jaccard.png
    fig3_vote_distribution.png    fig6_pca_variance.png
    stats.json                    routes_scored.csv

Requirements:
    pip install pandas numpy scikit-learn matplotlib
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# ----------------------------------------------------------------------
# CONFIG

# ----------------------------------------------------------------------
RAW_CSV = os.path.expanduser("~/Downloads/Flights1_2019_1 (1).csv")
OUT_DIR = os.path.expanduser("~/Downloads/figures")

MIN_FLIGHTS = 5       # routes with fewer flights are excluded (unreliable aggregates)
CONTAMINATION = 0.05  # expected anomaly fraction, shared by all models
N_PCA = 3
SEED = 42

# A route's "probability of delay" = share of its flights that were LATE (delay > 0).
# Set to False to instead measure the share that departed/arrived EARLY (delay < 0).
DELAY_MEANS_LATE = True

FEATURES = [
    "number_of_flights", "mean_departure_delay", "probablility_of_dep_delay",
    "max_departure_delay", "mean_arrival_delay", "probablility_of_arr_delay",
    "max_arrival_delay",
]

# Report palette
NAVY, AMBER, GRAY, RED = "#0b1f3a", "#f5b942", "#c5cede", "#c0392b"
plt.rcParams.update({
    "figure.dpi": 150, "axes.edgecolor": NAVY, "axes.labelcolor": NAVY,
    "xtick.color": NAVY, "ytick.color": NAVY, "axes.titlecolor": NAVY,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
})



# 1. LOADING DATA AND AGGREGATE TO ROUTE LEVEL
# ----------------------------------------------------------------------
def build_routes():
    df = pd.read_csv(RAW_CSV, low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    n_raw = len(df)

    # No CANCELLED column in this extract — missing delays are the signal.
    df = df.dropna(subset=["DEP_DELAY", "ARR_DELAY"])
    df["flight_path"] = df["ORIGIN_CITY_NAME"] + " -> " + df["DEST_CITY_NAME"]

    def share(x, col):
        return (x[col] > 0).mean() if DELAY_MEANS_LATE else (x[col] < 0).mean()

    g = df.groupby("flight_path")
    routes = pd.DataFrame({
        "number_of_flights": g["ARR_DELAY"].count(),
        "mean_departure_delay": g["DEP_DELAY"].mean(),
        "probablility_of_dep_delay": g.apply(share, col="DEP_DELAY", include_groups=False),
        "max_departure_delay": g["DEP_DELAY"].max(),
        "mean_arrival_delay": g["ARR_DELAY"].mean(),
        "probablility_of_arr_delay": g.apply(share, col="ARR_DELAY", include_groups=False),
        "max_arrival_delay": g["ARR_DELAY"].max(),
    }).reset_index()

    meta = {"n_raw": n_raw, "n_clean": len(df), "n_routes_all": len(routes),
            "pct_flights_late": float((df["ARR_DELAY"] > 0).mean()),
            "mean_arr_delay_flights": float(df["ARR_DELAY"].mean())}
    return routes, meta



# 2. PCA + 5 DETECTORS + CONSENSUS VOTE
# ----------------------------------------------------------------------
def detect(routes):
    routes = routes[routes["number_of_flights"] >= MIN_FLIGHTS].reset_index(drop=True)
    X = StandardScaler().fit_transform(routes[FEATURES].values)

    pca_full = PCA(n_components=len(FEATURES), random_state=SEED).fit(X)
    coords = PCA(n_components=N_PCA, random_state=SEED).fit_transform(X)

    iso = IsolationForest(contamination=CONTAMINATION, random_state=SEED, n_estimators=200)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION, novelty=True).fit(X)
    ocs = OneClassSVM(nu=CONTAMINATION, kernel="rbf", gamma="scale").fit(X)
    ell = EllipticEnvelope(contamination=CONTAMINATION, random_state=SEED)
    dbs = DBSCAN(eps=1.2, min_samples=5).fit_predict(X)

    models = {
        "Isolation Forest": (iso.fit_predict(X), -iso.score_samples(X)),
        "Local Outlier Factor": (lof.predict(X), -lof.score_samples(X)),
        "One-Class SVM": (ocs.predict(X), -ocs.decision_function(X)),
        "Elliptic Envelope": (ell.fit_predict(X), -ell.score_samples(X)),
        "DBSCAN": (np.where(dbs == -1, -1, 1), (dbs == -1).astype(float)),
    }

    for name, (pred, score) in models.items():
        routes[f"flag_{name}"] = pred == -1
        routes[f"score_{name}"] = (score - score.min()) / (score.max() - score.min() + 1e-9)

    algos = list(models)
    routes["votes"] = routes[[f"flag_{a}" for a in algos]].sum(axis=1)
    routes["consensus"] = routes["votes"] >= 2
    return routes, X, coords, pca_full.explained_variance_ratio_, algos



# 3. visualizations 
# ----------------------------------------------------------------------
def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name))
    plt.close(fig)
    print(f"  wrote {name}")


def make_figures(routes, coords, ev, algos, jaccard):
    n = len(routes)

    # Fig 1 — distribution of mean arrival delay
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.hist(routes["mean_arrival_delay"], bins=70, color=NAVY, edgecolor="white", linewidth=.3)
    mean = routes["mean_arrival_delay"].mean()
    ax.axvline(mean, color=AMBER, ls="--", lw=2, label=f"mean = {mean:.1f} min")
    ax.set(xlabel="Mean arrival delay (min)", ylabel="Routes",
           title=f"Mean arrival delay across {n:,} routes — January 2019")
    ax.legend(frameon=False)
    save(fig, "fig1_delay_distribution.png")

    # Fig 2 — PCA scatter with consensus anomalies
    m = routes["consensus"].values
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.scatter(coords[~m, 0], coords[~m, 1], s=8, c=GRAY, alpha=.55, label=f"Normal ({(~m).sum():,})")
    ax.scatter(coords[m, 0], coords[m, 1], s=16, c=RED, alpha=.85, label=f"Consensus anomalies ({m.sum():,})")
    ax.set(xlabel="PC1", ylabel="PC2", title="Routes in PCA space — consensus anomalies (2+ votes) in red")
    ax.legend(frameon=False)
    save(fig, "fig2_pca_anomalies.png")

    # Fig 3 — vote distribution
    vd = routes["votes"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    colors = [GRAY if k < 2 else (AMBER if k < 4 else RED) for k in vd.index]
    bars = ax.bar(vd.index.astype(str), vd.values, color=colors, edgecolor=NAVY, linewidth=.5)
    for bar, v in zip(bars, vd.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + n * .008, f"{v:,}", ha="center", fontsize=8, color=NAVY)
    ax.set(xlabel="Number of algorithms flagging the route (votes)", ylabel="Routes",
           title=f"Consensus voting — {int(m.sum()):,} routes flagged by 2+ of {len(algos)} algorithms")
    save(fig, "fig3_vote_distribution.png")

    # Fig 4 — unanimous anomalies by mean arrival delay
    top = routes[routes["votes"] == len(algos)].nlargest(12, "mean_arrival_delay")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(top["flight_path"], top["mean_arrival_delay"], color=NAVY, edgecolor="white")
    ax.invert_yaxis()
    ax.set(xlabel="Mean arrival delay (min)",
           title=f"Unanimous anomalies ({len(algos)}/{len(algos)} votes) — worst mean arrival delay")
    save(fig, "fig4_top_anomalies.png")

    # Fig 5 — Jaccard agreement heatmap
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    im = ax.imshow(jaccard.values, cmap="cividis", vmin=0, vmax=1)
    ax.set_xticks(range(len(algos)), algos, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(algos)), algos, fontsize=8)
    for i in range(len(algos)):
        for j in range(len(algos)):
            v = jaccard.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v < .6 else NAVY)
    ax.set_title("Cross-model agreement — Jaccard similarity of flagged sets")
    fig.colorbar(im, shrink=.8)
    save(fig, "fig5_jaccard.png")

    # Fig 6 — explained variance
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.bar([f"PC{i+1}" for i in range(len(ev))], ev, color=NAVY, edgecolor="white")
    ax.plot(range(len(ev)), np.cumsum(ev), color=AMBER, marker="o", lw=2, label="cumulative")
    ax.axhline(.9, color=RED, ls="--", lw=1, label="90% threshold")
    ax.set(ylabel="Explained variance ratio",
           title=f"PCA — {N_PCA} components capture {ev[:N_PCA].sum():.1%} of route-level variance")
    ax.legend(frameon=False)
    save(fig, "fig6_pca_variance.png")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Aggregating flights to route level...")
    routes, meta = build_routes()

    print("Running PCA and 5 anomaly detectors...")
    routes, X, coords, ev, algos = detect(routes)

    jaccard = pd.DataFrame(index=algos, columns=algos, dtype=float)
    for a in algos:
        for b in algos:
            sa, sb = set(routes.index[routes[f"flag_{a}"]]), set(routes.index[routes[f"flag_{b}"]])
            jaccard.loc[a, b] = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

    print("Drawing figures...")
    make_figures(routes, coords, ev, algos, jaccard)

    means = routes.groupby("consensus")[FEATURES].mean()
    stats = {
        **meta,
        "n_routes_analyzed": len(routes),
        "explained_variance": ev.round(4).tolist(),
        f"cumulative_variance_{N_PCA}pc": float(ev[:N_PCA].sum()),
        "flags_per_algorithm": {a: int(routes[f"flag_{a}"].sum()) for a in algos},
        "n_consensus": int(routes["consensus"].sum()),
        "n_unanimous": int((routes["votes"] == len(algos)).sum()),
        "vote_distribution": routes["votes"].value_counts().sort_index().to_dict(),
        "silhouette": float(silhouette_score(X, routes["consensus"].astype(int))),
        "jaccard": jaccard.round(3).to_dict(),
        "spearman": routes[[f"score_{a}" for a in algos]].corr(method="spearman").round(3).values.tolist(),
        "feature_means_normal": means.loc[False].round(2).to_dict(),
        "feature_means_anomaly": means.loc[True].round(2).to_dict(),
    }
    with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    routes.to_csv(os.path.join(OUT_DIR, "routes_scored.csv"), index=False)

    print(f"\nDone. {len(routes):,} routes analyzed | "
          f"{stats['n_consensus']:,} consensus anomalies | "
          f"{stats['n_unanimous']} unanimous | silhouette {stats['silhouette']:.3f}")
    print(f"Figures and stats.json in {OUT_DIR}/")


if __name__ == "__main__":
    main()
