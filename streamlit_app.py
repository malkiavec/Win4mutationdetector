
# app.py
# Modular Lottery Prediction Dashboard with API-ready data handler,
# feature engineering, model trainer, and visualization.
# Includes Pick 4–aware logic, positionless overlap learning, and boost persistence.

import os
import io
import json
import math
import random
from dataclasses import dataclass
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

# Optional plotting (Plotly fallback if seaborn/matplotlib missing)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="Lottery Prediction Dashboard", page_icon="🎲", layout="wide")
st.title("🎲 Lottery Prediction Dashboard (Pick 4)")

STATE_DIR = ".p4_dashboard_state"
os.makedirs(STATE_DIR, exist_ok=True)
BOOST_PATH = os.path.join(STATE_DIR, "boost_counts.json")


# ---------------------------
# Utility functions
# ---------------------------
def to_digits(x, n=4) -> Tuple[int, ...]:
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        digs = [int(v) for v in x]
    else:
        s = "".join(ch for ch in str(x) if ch.isdigit())
        digs = [int(ch) for ch in s]
    if len(digs) > n:
        digs = digs[-n:]
    elif len(digs) < n:
        digs = [0] * (n - len(digs)) + digs
    return tuple(digs)

def digits_to_str(t: Tuple[int, ...]) -> str:
    return "".join(str(int(d)) for d in t)

def multiset_overlap(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    ca, cb = Counter(a), Counter(b)
    return sum(min(ca[k], cb[k]) for k in ca.keys() | cb.keys())

def mod10(x: int) -> int:
    return x % 10

def greedy_multiset_mapping(a: Tuple[int, ...], b: Tuple[int, ...]) -> List[Tuple[int, int]]:
    ca, cb = Counter(a), Counter(b)
    pairs: List[Tuple[int, int]] = []
    for d in range(10):
        m = min(ca[d], cb[d])
        if m:
            pairs.extend((d, d) for _ in range(m))
            ca[d] -= m
            cb[d] -= m
    rem_a, rem_b = [], []
    for d in range(10):
        if ca[d] > 0: rem_a.extend([d] * ca[d])
        if cb[d] > 0: rem_b.extend([d] * cb[d])
    rem_a.sort()
    rem_b.sort()
    pairs.extend(zip(rem_a, rem_b))
    return pairs

def load_boosts() -> Dict[str, float]:
    if os.path.exists(BOOST_PATH):
        try:
            with open(BOOST_PATH, "r") as f:
                raw = json.load(f)
            return {k: float(v) for k, v in raw.items()}
        except Exception:
            return {}
    return {}

def save_boosts(boosts: Dict[str, float]) -> None:
    try:
        with open(BOOST_PATH, "w") as f:
            json.dump(boosts, f)
    except Exception:
        pass


# ---------------------------
# API Client (stub)
# ---------------------------
@dataclass
class LotteryAPI:
    # Replace base_url and auth configuration for real API
    base_url: str = "https://data.ny.gov/resource/hsys-3def.json"
    game: str = "pick4"

    def fetch_recent_draws(self, limit: int = 500) -> pd.DataFrame:
        # Stubbed demo: generate synthetic pick-4 draws
        # Replace with: requests.get(...), parse JSON -> DataFrame
        rng = np.random.default_rng(seed=42)
        rows = []
        date = pd.Timestamp.utcnow().normalize()
        cur = (1, 2, 3, 4)
        for i in range(limit):
            # simple evolution
            shifts = rng.choice([-2, -1, 0, 1, 2], size=4)
            cur = tuple(mod10(cur[j] + int(shifts[j])) for j in range(4))
            rows.append(
                {
                    "draw_date": date - pd.Timedelta(days=limit - i),
                    "winning_numbers": cur,
                    "sum": int(sum(cur)),
                }
            )
        return pd.DataFrame(rows)


# ---------------------------
# Data Handler
# ---------------------------
class LotteryDataHandler:
    def __init__(self, api: Optional[LotteryAPI] = None, n_digits: int = 4):
        self.api = api or LotteryAPI()
        self.n_digits = n_digits

    @st.cache_data(show_spinner=False)
    def get_processed_data(self, use_api: bool, uploaded_csv: Optional[io.BytesIO]) -> pd.DataFrame:
        if use_api:
            df = self.api.fetch_recent_draws(limit=800)
        else:
            if uploaded_csv is None:
                # Minimal sample
                df = pd.DataFrame({"draw_date": pd.date_range(end=pd.Timestamp.utcnow(), periods=15),
                                   "winning_numbers": [(1,2,3,4),(3,6,0,1),(9,8,7,0),(7,4,1,2),(2,5,8,3),
                                                       (1,2,3,5),(3,6,9,1),(0,2,4,6),(1,3,5,7),(2,4,6,8),
                                                       (3,5,7,9),(4,6,8,0),(5,7,9,1),(6,9,1,3),(8,0,3,5)]})
                df["sum"] = df["winning_numbers"].apply(sum)
            else:
                raw = pd.read_csv(uploaded_csv)
                # Try to infer columns
                if "winning_numbers" in raw.columns and isinstance(raw["winning_numbers"].iloc[0], str):
                    def parse_tuple(s):
                        digs = [int(ch) for ch in s if ch.isdigit()]
                        return tuple(digs[:self.n_digits]) if len(digs) >= self.n_digits else to_digits(digs, self.n_digits)
                    wn = raw["winning_numbers"].apply(parse_tuple)
                else:
                    # Assume first column has draws like 4728
                    wn = raw.iloc[:, 0].apply(lambda x: to_digits(x, self.n_digits))
                if "draw_date" in raw.columns:
                    dt = pd.to_datetime(raw["draw_date"], errors="coerce")
                else:
                    dt = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(raw))
                df = pd.DataFrame({"draw_date": dt, "winning_numbers": wn})
                df["sum"] = df["winning_numbers"].apply(sum)

        df = df.dropna(subset=["draw_date"]).sort_values("draw_date").reset_index(drop=True)
        return df


# ---------------------------
# Feature Engineering
# ---------------------------
class FeatureEngineer:
    def __init__(self, n_digits: int = 4):
        self.n_digits = n_digits
        self.fitted_ = False
        self.means_ = None
        self.stds_ = None

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Expand digits and simple stats
        digs = pd.DataFrame(df["winning_numbers"].tolist(), columns=[f"d{i+1}" for i in range(self.n_digits)])
        out = pd.DataFrame()
        out[[f"d{i+1}" for i in range(self.n_digits)]] = digs
        out["sum"] = digs.sum(axis=1)
        out["odd_count"] = digs.apply(lambda r: sum(1 for v in r if v % 2 == 1), axis=1)
        out["even_count"] = self.n_digits - out["odd_count"]
        out["unique_count"] = digs.apply(lambda r: len(set(r)), axis=1)
        out["repeat_pair"] = digs.apply(lambda r: int(any(c >= 2 for c in Counter(r).values())), axis=1)
        # Rolling features (requires order)
        out["sum_rolling_mean_10"] = out["sum"].rolling(10, min_periods=1).mean()
        out["sum_rolling_std_10"] = out["sum"].rolling(10, min_periods=1).std().fillna(0.0)
        return out

    def prepare_data(self, df: pd.DataFrame, history_window: int = 600) -> Tuple[pd.DataFrame, pd.DataFrame]:
        feats = self.create_features(df)
        # Train split: all but last row; test: last row (for live prediction demo)
        X_train = feats.iloc[:-1].reset_index(drop=True)
        X_test = feats.iloc[[-1]].reset_index(drop=True)
        self.fit_scaler(X_train)
        return self.transform_features(X_train), self.transform_features(X_test)

    def fit_scaler(self, X: pd.DataFrame):
        self.means_ = X.mean()
        self.stds_ = X.std().replace(0, 1.0)
        self.fitted_ = True

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            self.fit_scaler(X)
        return (X - self.means_) / self.stds_


# ---------------------------
# Model Trainer (hybrid)
# ---------------------------
class LotteryModelTrainer:
    def __init__(self, n_digits: int = 4):
        self.n_digits = n_digits
        # Simple frequency and transition storage
        self.digit_freq_ = np.ones(10)  # Laplace
        self.trans_counts_ = Counter()
        self.alpha_ = 0.5  # smoothing

    def fit(self, history_numbers: List[Tuple[int, ...]]):
        # Digit frequency
        freq = Counter()
        for t in history_numbers:
            for d in t:
                freq[d] += 1
        self.digit_freq_ = np.array([freq.get(i, 0) + 1.0 for i in range(10)], dtype=float)
        self.digit_freq_ = self.digit_freq_ / self.digit_freq_.sum()

        # Positionless digit transitions (lag=1)
        self.trans_counts_ = Counter()
        for i in range(len(history_numbers) - 1):
            a, b = history_numbers[i], history_numbers[i + 1]
            for x, y in greedy_multiset_mapping(a, b):
                self.trans_counts_[(x, y)] += 1

    def transition_probs(self, boosts: Optional[Dict[str, float]] = None, boost_weight: float = 1.0) -> Dict[Tuple[int, int], float]:
        cnt = self.trans_counts_.copy()
        totals = Counter()
        for (x, y), c in cnt.items():
            totals[x] += c
        if boosts:
            for k, v in boosts.items():
                try:
                    xs, ys = k.split("->")
                    x, y = int(xs), int(ys)
                    cnt[(x, y)] += float(v) * boost_weight
                    totals[x] += float(v) * boost_weight
                except Exception:
                    continue
        probs: Dict[Tuple[int, int], float] = {}
        alpha = self.alpha_
        for x in range(10):
            denom = totals[x] + alpha * 10
            if denom <= 0:
                for y in range(10):
                    probs[(x, y)] = 0.1
                continue
            for y in range(10):
                c = cnt.get((x, y), 0.0)
                probs[(x, y)] = (c + alpha) / denom
        return probs

    def base_predict_distribution(self, last_draw: Tuple[int, ...], boosts: Optional[Dict[str, float]] = None, boost_weight: float = 1.0) -> List[Tuple[int, float]]:
        # For each digit in last_draw, pick top candidates and combine
        probs = self.transition_probs(boosts=boosts, boost_weight=boost_weight)
        per_digit = []
        for d in last_draw:
            row = [(y, probs.get((d, y), 1e-3)) for y in range(10)]
            row.sort(key=lambda t: t[1], reverse=True)
            per_digit.append(row[:4])  # top-4 per digit

        # Combine into candidate tuples with product probability
        cands: Dict[Tuple[int, ...], float] = {}
        def dfs(i, cur, p):
            if i == len(per_digit):
                t = tuple(cur)
                cands[t] = cands.get(t, 0.0) + p
                return
            for y, py in per_digit[i]:
                cur.append(y)
                dfs(i + 1, cur, p * py)
                cur.pop()
        dfs(0, [], 1.0)
        # Normalize
        total_p = sum(cands.values()) or 1.0
        ranked = sorted(((t, v / total_p) for t, v in cands.items()), key=lambda kv: kv[1], reverse=True)
        return ranked

    def mutate(self, t: Tuple[int, ...], strength: float) -> Tuple[int, ...]:
        # Mutation: randomly shift each digit by small step based on strength
        max_step = max(1, int(5 * strength))
        return tuple(mod10(d + random.randint(-max_step, max_step)) for d in t)

    def generate_predictions(self, last_draw: Tuple[int, ...], n: int, mutation_strength: float,
                             boosts: Optional[Dict[str, float]] = None, boost_weight: float = 1.0) -> List[Tuple[int, ...]]:
        ranked = self.base_predict_distribution(last_draw, boosts=boosts, boost_weight=boost_weight)
        seeds = [t for t, _ in ranked[:max(5, n//2)]]
        preds = set(seeds)
        while len(preds) < n:
            base = random.choice(seeds)
            preds.add(self.mutate(base, mutation_strength))
        return list(preds)[:n]


# ---------------------------
# Visualization
# ---------------------------
class LotteryVisualizer:
    def plot_winning_numbers_trend(self, df: pd.DataFrame):
        if not HAS_PLOTLY:
            return st.write("Plotly not available.")
        vals = pd.DataFrame(df["winning_numbers"].tolist(), columns=[f"d{i+1}" for i in range(4)])
        vals["draw_date"] = df["draw_date"].values
        long = vals.melt(id_vars="draw_date", var_name="digit_pos", value_name="value")
        fig = px.line(long, x="draw_date", y="value", color="digit_pos", title="Digit values over time")
        return fig

    def plot_sum_distribution(self, df: pd.DataFrame):
        if not HAS_PLOTLY:
            return st.write("Plotly not available.")
        fig = px.histogram(df, x="sum", nbins=20, title="Sum distribution")
        return fig

    def plot_model_comparison(self, metrics: Dict[str, float]):
        if not HAS_PLOTLY:
            return st.write(metrics)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(metrics.keys()), y=list(metrics.values())))
        fig.update_layout(title="Model comparison (higher is better)", yaxis_title="Score")
        return fig

    def plot_feature_importance(self, importances: List[float], feature_names: List[str]):
        if not HAS_PLOTLY:
            return st.write("Feature importances:", dict(zip(feature_names, importances)))
        fig = go.Figure([go.Bar(x=feature_names, y=importances)])
        fig.update_layout(title="Feature importance")
        return fig


# ---------------------------
# App sidebar
# ---------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Historical Analysis", "Predictions", "Model Performance"], index=0)

st.sidebar.header("Data source")
use_api = st.sidebar.toggle("Use API (demo)", value=False)
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

st.sidebar.header("Prediction settings")
mutation_strength = st.sidebar.slider("Mutation Strength", 0.0, 1.0, 0.4, 0.05)
num_predictions = st.sidebar.select_slider("Number of predictions", options=[10, 20, 30, 50, 100], value=30)

st.sidebar.header("Learning/Feedback")
enable_boosts = st.sidebar.checkbox("Enable feedback boosts", value=False)
boost_weight = st.sidebar.slider("Boost weight", 0.0, 5.0, 1.0, 0.1)
success_threshold = st.sidebar.selectbox("Success threshold (positionless overlap)", options=[3, 4], index=0)

# ---------------------------
# Initialize components
# ---------------------------
api = LotteryAPI()
data_handler = LotteryDataHandler(api=api)
feature_engineer = FeatureEngineer()
visualizer = LotteryVisualizer()
model_trainer = LotteryModelTrainer()

@st.cache_data(show_spinner=False)
def load_data_cached(use_api_flag: bool, uploaded_csv_bytes: Optional[bytes]) -> pd.DataFrame:
    bio = io.BytesIO(uploaded_csv_bytes) if uploaded_csv_bytes else None
    return data_handler.get_processed_data(use_api_flag, bio)

# Load data
try:
    uploaded_bytes = uploaded.read() if uploaded else None
    data = load_data_cached(use_api, uploaded_bytes)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Fit model on all but last row
numbers_list = [to_digits(t, 4) for t in data["winning_numbers"].tolist()]
if len(numbers_list) < 5:
    st.warning("Limited history; results may be noisy.")
model_trainer.fit(numbers_list[:-1] if len(numbers_list) > 1 else numbers_list)

# ---------------------------
# Pages
# ---------------------------
if page == "Historical Analysis":
    st.header("Historical Lottery Data Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(visualizer.plot_winning_numbers_trend(data), use_container_width=True) if HAS_PLOTLY else st.write("Install plotly for charts.")
    with c2:
        st.plotly_chart(visualizer.plot_sum_distribution(data), use_container_width=True) if HAS_PLOTLY else st.write("Install plotly for charts.")
    st.subheader("Recent Drawing History")
    df_show = data[["draw_date", "winning_numbers", "sum"]].tail(15).sort_values("draw_date", ascending=False)
    st.dataframe(df_show, use_container_width=True)

elif page == "Predictions":
    st.header("Lottery Number Predictions")
    latest_numbers = numbers_list[-1] if numbers_list else (1, 2, 3, 4)
    st.subheader("Most Recent Winning Numbers")
    st.write(f"Draw Date: {data.iloc[-1]['draw_date'].strftime('%Y-%m-%d')}")
    st.write("Numbers:", latest_numbers)
    st.write("Sum:", sum(latest_numbers))

    if st.button("Generate Predictions", type="primary"):
        with st.spinner("Generating predictions..."):
            boosts = load_boosts() if enable_boosts else {}
            preds = model_trainer.generate_predictions(
                last_draw=latest_numbers,
                n=num_predictions,
                mutation_strength=mutation_strength,
                boosts=boosts,
                boost_weight=boost_weight,
            )
            preds_str = [digits_to_str(p) for p in preds]
            st.success("Predictions generated.")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("Top predictions")
                st.write(preds_str)

            with col2:
                st.markdown("Teach the model (manual success marking)")
                seed_input = st.text_input("Seed (previous draw)", value=digits_to_str(numbers_list[-2]) if len(numbers_list) >= 2 else "")
                actual_input = st.text_input("Actual next draw", value="")
                use_current_preds = st.checkbox("Use current predictions", value=True)
                pasted_preds = st.text_area("Or paste predictions (comma-separated)", value="")
                if st.button("Mark success (positionless)"):
                    seed_t = to_digits(seed_input, 4)
                    actual_t = to_digits(actual_input, 4)
                    pred_list = preds_str if use_current_preds else [p.strip() for p in pasted_preds.split(",") if p.strip()]
                    if not pred_list:
                        st.warning("No predictions provided.")
                    else:
                        overlaps = [multiset_overlap(to_digits(p, 4), actual_t) for p in pred_list]
                        best_overlap = max(overlaps)
                        st.write(f"Best overlap: {best_overlap}")
                        if enable_boosts and best_overlap >= success_threshold:
                            # Boost using the best prediction
                            best_idx = int(np.argmax(overlaps))
                            best_pred_t = to_digits(pred_list[best_idx], 4)
                            pairs = greedy_multiset_mapping(seed_t, best_pred_t)
                            b = load_boosts()
                            for x, y in pairs:
                                key = f"{x}->{y}"
                                b[key] = b.get(key, 0.0) + 1.0
                            save_boosts(b)
                            st.success("Success recorded and boosts updated.")
                        else:
                            st.info("Success recorded (no boosts applied or below threshold).")

            # Export
            st.download_button("Download predictions (CSV)",
                               pd.DataFrame({"prediction": preds_str}).to_csv(index=False).encode("utf-8"),
                               file_name="predictions.csv",
                               mime="text/csv")

else:
    st.header("Model Performance")
    # Simple evaluation proxy: rolling 50-window 3-of-4 hit rate on historical one-step-ahead
    rows = []
    boosts = load_boosts() if enable_boosts else {}
    for t in range(1, len(numbers_list)):
        hist = numbers_list[:t]
        model_trainer.fit(hist[:-1] if len(hist) > 1 else hist)
        last = hist[-1]
        preds = model_trainer.generate_predictions(last, n=30, mutation_strength=0.3, boosts=boosts, boost_weight=boost_weight)
        actual = numbers_list[t]
        best_overlap = max(multiset_overlap(p, actual) for p in preds) if preds else 0
        rows.append({"t": t, "best_overlap": best_overlap, "success3": int(best_overlap >= 3), "success4": int(best_overlap >= 4)})
    perf = pd.DataFrame(rows)
    if perf.empty:
        st.info("Not enough data for evaluation.")
    else:
        perf["rolling_success3"] = perf["success3"].rolling(50, min_periods=1).mean()
        perf["rolling_success4"] = perf["success4"].rolling(50, min_periods=1).mean()
        st.dataframe(perf.tail(20), use_container_width=True)
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=perf["t"], y=perf["rolling_success3"], name="Rolling success ≥3"))
            fig.add_trace(go.Scatter(x=perf["t"], y=perf["rolling_success4"], name="Rolling success =4"))
            fig.update_layout(title="Rolling success rates", xaxis_title="t (step)", yaxis_title="Rate")
            st.plotly_chart(fig, use_container_width=True)

    # Show current boosts and a clear button
    st.subheader("Boosts status")
    current_boosts = load_boosts()
    st.write(f"Stored boosts: {sum(current_boosts.values())} total pseudo-counts across {len(current_boosts)} transitions")
    if st.button("Clear boosts"):
        save_boosts({})
        st.success("Boosts cleared.")
