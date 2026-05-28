from __future__ import annotations


"""

0_4_StatsRerank_D.py — Loto 7/39 NEXT predikcija

Mod D = B + C:
  soft statistički rerank + per-pozicija prior.

U fajlu su tri pojedinačna regresora:
  1. DTR  = DecisionTreeRegressor
  2. RFR  = RandomForestRegressor
  3. XGB  = XGBRegressor + MultiOutputRegressor

Svaki regresor prvo daje osnovne skorove za brojeve 1..39.
Zatim se iz top kandidata generišu kombinacije i rerankuju po formuli:

  total_score = model_score
              + STATS_WEIGHT * stats_score
              + POSITION_WEIGHT * position_score

stats_score nagrađuje kombinacije bliže istorijskim statistikama:
  suma, broj neparnih, broj niskih (<=19), raspon

position_score koristi istorijsku verovatnoću:
  P(broj | sortirana pozicija 0..6)

"""


import os

SEED = 39
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import itertools
import random
import time
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import label_ranking_average_precision_score, roc_auc_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    import xgboost as xgb
except Exception as e:
    raise RuntimeError("Nedostaje xgboost — pokreni: pip install xgboost") from e

try:
    from qiskit_machine_learning.utils import algorithm_globals
    algorithm_globals.random_seed = SEED
except Exception:
    pass

warnings.filterwarnings("ignore")
np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# Konfiguracija
# ============================================================
CSV_PATH = "/data/loto7_4622_k42.csv"
OUT_TXT = "/0_4_StatsRerank_D_predikcija.txt"
PLOT_PATH = "/0_4_SkewnessKurtosis_D.png"

N_MIN, N_MAX = 1, 39
K = 7
LAG = 5
WINDOWS = (20, 50, 100)
BACKTEST_N = 100
TOP_POOL = 15
STATS_WEIGHT = 0.25
POSITION_WEIGHT = 0.25
SMOOTH = 1.0


def stamp() -> str:
    return datetime.now(pytz.timezone("Europe/Belgrade")).strftime("%d.%m.%Y_%H.%M.%S")


T0 = time.time()
print()
print("🔁 0_4_StatsRerank_D — start ", stamp())
print()


# ============================================================
# 1) Učitavanje CSV-a
# ============================================================
df = pd.read_csv(CSV_PATH, header=None).iloc[:, :K].astype(int)
draws = np.sort(df.values, axis=1)
N = draws.shape[0]

print(f"✅ CSV učitan: {CSV_PATH}")
print(f"   broj izvlačenja: {N}, brojeva po kolu: {K}")
print()


def draws_to_multihot(rows: np.ndarray) -> np.ndarray:
    out = np.zeros((rows.shape[0], N_MAX), dtype=np.int8)
    for i, row in enumerate(rows):
        for v in row:
            if N_MIN <= v <= N_MAX:
                out[i, v - 1] = 1
    return out


Y_full = draws_to_multihot(draws)


# ============================================================
# 2) Skew/kurt plot + statistički i pozicioni prior
# ============================================================
stats_df = pd.DataFrame({
    "suma": draws.sum(axis=1),
    "neparnih": (draws % 2 == 1).sum(axis=1),
    "niskih": (draws <= 19).sum(axis=1),
    "raspon": draws.max(axis=1) - draws.min(axis=1),
}).astype(float)

stats_params = {}
for col in stats_df.columns:
    mu = float(stats_df[col].mean())
    sd = float(stats_df[col].std(ddof=0))
    stats_params[col] = (mu, max(sd, 1e-9))

position_counts = np.full((K, N_MAX), SMOOTH, dtype=float)
for row in draws:
    sorted_row = np.sort(row)
    for pos, value in enumerate(sorted_row):
        position_counts[pos, value - 1] += 1.0
position_probs = position_counts / position_counts.sum(axis=1, keepdims=True)
position_log_probs = np.log(position_probs)

print("📐 Mod D = statistički prior + per-pozicija prior")
for col, (mu, sd) in stats_params.items():
    print(f"   {col:<8} mean={mu:7.3f}, std={sd:6.3f}")
print()

fig, axes = plt.subplots(len(stats_df.columns), 3, figsize=(15, 4 * len(stats_df.columns)))
for i, col in enumerate(stats_df.columns):
    series = stats_df[col].dropna()
    series.hist(ax=axes[i, 0], bins=30, color="steelblue", edgecolor="white")
    axes[i, 0].set_title(f"{col} — Histogram")

    axes[i, 1].boxplot(series.values, vert=True)
    axes[i, 1].set_title(f"{col} — Box Plot")
    axes[i, 1].set_xticks([1], [col])

    scipy_stats.probplot(series.values, dist="norm", plot=axes[i, 2])
    axes[i, 2].set_title(f"{col} — QQ Plot")

plt.tight_layout()
plt.savefig(PLOT_PATH)


# ============================================================
# 3) Feature engineering
# ============================================================
def build_features(draws_arr: np.ndarray,
                   y_multi: np.ndarray,
                   lag: int = LAG,
                   windows=WINDOWS) -> np.ndarray:
    n, _ = draws_arr.shape

    lag_feats = []
    for L in range(1, lag + 1):
        shifted = np.zeros_like(draws_arr)
        shifted[L:] = draws_arr[:-L]
        lag_feats.append(shifted)
    lag_block = np.concatenate(lag_feats, axis=1)

    cum = np.cumsum(y_multi, axis=0)
    rolling_blocks = []
    for W in windows:
        rolled = np.zeros_like(cum, dtype=float)
        rolled[1:W + 1] = cum[:W]
        rolled[W + 1:] = cum[W:-1] - cum[:-W - 1]
        rolling_blocks.append(rolled / float(W))
    roll_block = np.concatenate(rolling_blocks, axis=1)

    gap = np.zeros((n, N_MAX), dtype=float)
    last_seen = np.full(N_MAX, -1, dtype=int)
    for i in range(n):
        for k in range(N_MAX):
            gap[i, k] = (i - last_seen[k]) if last_seen[k] >= 0 else i + 1
        for v in draws_arr[i]:
            last_seen[v - 1] = i

    prev = np.zeros_like(draws_arr)
    prev[1:] = draws_arr[:-1]
    s_sum = prev.sum(axis=1, keepdims=True).astype(float)
    s_odd = (prev % 2 == 1).sum(axis=1, keepdims=True).astype(float)
    s_low = (prev <= 19).sum(axis=1, keepdims=True).astype(float)
    s_rng = (prev.max(axis=1, keepdims=True) - prev.min(axis=1, keepdims=True)).astype(float)
    stat_block = np.concatenate([s_sum, s_odd, s_low, s_rng], axis=1)

    return np.concatenate([lag_block, roll_block, gap, stat_block], axis=1)


X_full = build_features(draws, Y_full)
START = max(LAG, max(WINDOWS))

X_all = X_full[START:N].astype(float)
Y_all = Y_full[START:N].astype(float)

n_total = X_all.shape[0]
n_train = n_total - BACKTEST_N
assert n_train > 200, "Premalo podataka za back-test."

X_train, Y_train = X_all[:n_train], Y_all[:n_train]
X_back, Y_back = X_all[n_train:], Y_all[n_train:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_back_s = scaler.transform(X_back)
X_next_s = scaler.transform(X_full[N - 1:N].astype(float))


# ============================================================
# 4) Modeli
# ============================================================
models = {
    "DTR": DecisionTreeRegressor(random_state=SEED, max_depth=10, min_samples_leaf=4),
    "RFR": RandomForestRegressor(n_estimators=400, max_depth=10, random_state=SEED, n_jobs=1),
    "XGB": MultiOutputRegressor(
        xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
            tree_method="hist",
        )
    ),
}

print("⚛️ Treniranje DTR/RFR/XGB ...")
for name, model in models.items():
    model.fit(X_train_s, Y_train)
    print(f"   ✅ {name} treniran.")
print()


# ============================================================
# 5) Rerank D: soft statistika + per-pozicija prior
# ============================================================
def topk_from_scores(scores_1d: np.ndarray, k: int = K) -> np.ndarray:
    s = np.asarray(scores_1d, dtype=float).copy()
    order = np.lexsort((np.arange(N_MAX), -s))
    return np.sort(order[:k] + 1)


def combo_stats(combo: np.ndarray) -> dict[str, float]:
    return {
        "suma": float(combo.sum()),
        "neparnih": float((combo % 2 == 1).sum()),
        "niskih": float((combo <= 19).sum()),
        "raspon": float(combo.max() - combo.min()),
    }


def stats_soft_score(combo: np.ndarray) -> float:
    cs = combo_stats(combo)
    score = 0.0
    for key, value in cs.items():
        mu, sd = stats_params[key]
        z = (value - mu) / sd
        score -= z * z
    return float(score)


def position_score(combo: np.ndarray) -> float:
    combo = np.sort(combo)
    score = 0.0
    for pos, value in enumerate(combo):
        score += position_log_probs[pos, value - 1]
    return float(score)


def rerank_d(scores_1d: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    s = np.asarray(scores_1d, dtype=float)
    top_pool = np.lexsort((np.arange(N_MAX), -s))[:TOP_POOL] + 1

    best_combo = None
    best_total = -np.inf
    best_stats = -np.inf
    best_pos = -np.inf

    for combo_tuple in itertools.combinations(top_pool, K):
        combo = np.sort(np.asarray(combo_tuple, dtype=int))
        model_score = float(s[combo - 1].sum())
        stat_score = stats_soft_score(combo)
        pos_score = position_score(combo)
        total_score = model_score + STATS_WEIGHT * stat_score + POSITION_WEIGHT * pos_score
        if total_score > best_total:
            best_total = total_score
            best_stats = stat_score
            best_pos = pos_score
            best_combo = combo

    return best_combo, best_total, best_stats, best_pos


def avg_hits(scores_2d, Y, use_rerank: bool) -> float:
    h = 0
    for i in range(scores_2d.shape[0]):
        true_set = set(np.where(Y[i] == 1)[0] + 1)
        pred = rerank_d(scores_2d[i])[0] if use_rerank else topk_from_scores(scores_2d[i])
        h += len(true_set & set(pred.tolist()))
    return h / scores_2d.shape[0]


def safe_auc(Y, scores):
    try:
        return roc_auc_score(Y, scores, average="macro")
    except Exception:
        return float("nan")


def safe_lrap(Y, scores):
    try:
        return label_ranking_average_precision_score(Y.astype(int), scores)
    except Exception:
        return float("nan")


def describe(pick: np.ndarray) -> str:
    cs = combo_stats(pick)
    return (
        f"suma={int(cs['suma'])}, "
        f"neparnih={int(cs['neparnih'])}/{K}, "
        f"niskih(≤19)={int(cs['niskih'])}/{K}, "
        f"raspon={int(cs['raspon'])}"
    )


# ============================================================
# 6) Rezultati
# ============================================================
txt_lines = []

print("📊 Tabela rezultata — mod D (B + C)")
print(f"{'model':<5} {'raw_pick':<31} {'rerank_D':<31} {'raw_h':>6} {'D_h':>6} {'AUC':>7} {'LRAP':>7} {'stat':>8} {'pos':>8}")

for name, model in models.items():
    scores_back = model.predict(X_back_s)
    scores_next = model.predict(X_next_s)[0]

    raw_pick = topk_from_scores(scores_next)
    rerank_pick, total_score, stat_score, pos_score = rerank_d(scores_next)

    raw_h = avg_hits(scores_back, Y_back, use_rerank=False)
    rerank_h = avg_hits(scores_back, Y_back, use_rerank=True)
    auc = safe_auc(Y_back, scores_back)
    lrap = safe_lrap(Y_back, scores_back)

    print(
        f"{name:<5} {str(raw_pick.tolist()):<31} {str(rerank_pick.tolist()):<31} "
        f"{raw_h:>6.3f} {rerank_h:>6.3f} {auc:>7.3f} {lrap:>7.3f} "
        f"{stat_score:>8.3f} {pos_score:>8.3f}"
    )
    txt_lines.append(
        f"{name}: raw={raw_pick.tolist()} ({describe(raw_pick)}); "
        f"rerank_D={rerank_pick.tolist()} ({describe(rerank_pick)}); "
        f"raw_hits={raw_h:.3f}; rerank_hits={rerank_h:.3f}; "
        f"AUC={auc:.3f}; LRAP={lrap:.3f}; "
        f"stat_score={stat_score:.3f}; position_score={pos_score:.3f}; total_score={total_score:.3f}"
    )

print()
print(f"(slučajan baseline ≈ {7*7/39:.3f} hits/7)")
print()


# ============================================================
# 7) Snimanje
# ============================================================
elapsed = time.time() - T0
with open(OUT_TXT, "a", encoding="utf-8") as f:
    f.write(
        f"\n--- {stamp()} (seed={SEED}, N={N}, mode=D, top_pool={TOP_POOL}, "
        f"stats_weight={STATS_WEIGHT}, position_weight={POSITION_WEIGHT}) ---\n"
    )
    for line in txt_lines:
        f.write(line + "\n")
    f.write("stats_params=" + str({k: tuple(round(x, 4) for x in v) for k, v in stats_params.items()}) + "\n")
    f.write(f"plot={PLOT_PATH}\n")
    f.write(f"ukupno_vreme={str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)\n")

print(f"📝 Snimljeno u: {OUT_TXT}")
print(f"🖼️  Plot snimljen u: {PLOT_PATH}")
print()

print("🔁 0_4_StatsRerank_D — stop ", stamp())
print(f"⏱️  Ukupno vreme: {str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)")
print()

plt.show()


"""
🔁 0_4_StatsRerank_D — start  28.05.2026_11.18.40

✅ CSV učitan: /data/loto7_4622_k42.csv
   broj izvlačenja: 4622, brojeva po kolu: 7

📐 Mod D = statistički prior + per-pozicija prior
   suma     mean=140.510, std=27.641
   neparnih mean=  3.592, std= 1.200
   niskih   mean=  3.384, std= 1.218
   raspon   mean= 29.923, std= 5.167

⚛️ Treniranje DTR/RFR/XGB ...
   ✅ DTR treniran.
   ✅ RFR treniran.
   ✅ XGB treniran.

📊 Tabela rezultata — mod D (B + C)
model raw_pick                        rerank_D                         raw_h    D_h     AUC    LRAP     stat      pos
DTR   [8, 13, 16, 23, 31, 34, 37]     [1, 8, 13, 23, 29, 32, 34]       1.480  1.370   0.506   0.254   -0.570  -17.567
RFR   [7, 8, 23, 26, 27, 32, 35]      [3, 8, 15, 23, 29, 32, 34]       1.170  1.170   0.492   0.245   -0.274  -17.894
XGB   [2, 7, 23, 28, 30, 32, 37]      [2, 7, 13, 23, 28, 32, 37]       1.140  1.260   0.498   0.239   -1.183  -17.409

(slučajan baseline ≈ 1.256 hits/7)

📝 Snimljeno u: /0_4_StatsRerank_D_predikcija.txt
🖼️  Plot snimljen u: /0_4_SkewnessKurtosis_D.png

🔁 0_4_StatsRerank_D — stop  28.05.2026_11.20.18
⏱️  Ukupno vreme: 0:01:37  (97.8 s)
"""
