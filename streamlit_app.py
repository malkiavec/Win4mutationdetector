import io

import pandas as pd
import streamlit as st

from data_handler import LotteryDataHandler, LotteryAPI
from feature_engineering import FeatureEngineer
from model_trainer import LotteryModelTrainer
from visualization import LotteryVisualizer
from utils import to_digits, digits_to_str, multiset_overlap, load_boosts, save_boosts

st.set_page_config(page_title="Lottery Prediction Dashboard", page_icon="🎲", layout="wide")
st.title("🎲 Lottery Prediction Dashboard")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Historical Analysis", "Predictions", "Model Performance"])

st.sidebar.header("Data source")
use_api = st.sidebar.toggle("Use API (demo)", value=False)
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

st.sidebar.header("Prediction Settings")
mutation_strength = st.sidebar.slider("Mutation Strength", 0.0, 1.0, 0.5,
    help="Higher values create more varied predictions")
num_predictions = st.sidebar.select_slider("Number of predictions", options=[10, 20, 30, 50, 100], value=30)

st.sidebar.header("Learning / Feedback")
enable_boosts = st.sidebar.checkbox("Enable feedback boosts", value=True)
boost_weight = st.sidebar.slider("Boost weight (use during ranking)", 0.0, 5.0, 1.0, 0.1)
success_threshold = st.sidebar.selectbox("Success threshold (overlap, any order)", options=[3, 4], index=0)

# Initialize modules
api = LotteryAPI()
data_handler = LotteryDataHandler(api=api)
feature_engineer = FeatureEngineer()
visualizer = LotteryVisualizer()
model_trainer = LotteryModelTrainer()

@st.cache_data(show_spinner=False)
def load_data_cached(use_api_flag: bool, uploaded_csv_bytes: bytes) -> pd.DataFrame:
    return data_handler.get_processed_data(use_api_flag, uploaded_csv_bytes)

# Load data
try:
    uploaded_bytes = uploaded.read() if uploaded else None
    data = load_data_cached(use_api, uploaded_bytes if uploaded_bytes else None)
except Exception as e:
    st.error(f"An error occurred while loading data: {e}")
    st.stop()

# Prepare history and fit model (train on all but last)
numbers_list = [to_digits(t, 4) for t in data["winning_numbers"].tolist()]
if len(numbers_list) < 5:
    st.warning("Limited history; results may be noisy.")
model_trainer.fit(numbers_list[:-1] if len(numbers_list) > 1 else numbers_list)

# Pages
if page == "Historical Analysis":
    st.header("Historical Lottery Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            visualizer.plot_winning_numbers_trend(data),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            visualizer.plot_sum_distribution(data),
            use_container_width=True
        )

    st.subheader("Recent Drawing History")
    st.dataframe(
        data[["draw_date", "winning_numbers", "sum"]]
        .tail(15)
        .sort_values("draw_date", ascending=False),
        use_container_width=True
    )

elif page == "Predictions":
    st.header("Lottery Number Predictions")

    latest_numbers = numbers_list[-1] if numbers_list else (1, 2, 3, 4)
    st.subheader("Most Recent Winning Numbers")
    st.write(f"Draw Date: {data.iloc[-1]['draw_date'].strftime('%Y-%m-%d')}")
    st.write("Numbers:", latest_numbers)
    st.write("Sum:", sum(latest_numbers))

    if st.button("Generate Predictions", type="primary"):
        with st.spinner('Analyzing patterns and generating predictions...'):
            try:
                boosts = load_boosts() if enable_boosts else {}
                preds = model_trainer.generate_predictions(
                    last_draw=latest_numbers,
                    n=num_predictions,
                    mutation_strength=mutation_strength,
                    boosts=boosts,
                    boost_weight=boost_weight
                )
                preds_str = [digits_to_str(p) for p in preds]

                st.success("🎯 Predictions Generated!")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Top Predictions")
                    st.write(preds_str)

                with col2:
                    st.markdown("### Teach the model (manual success marking)")
                    seed_input = st.text_input("Seed (previous draw)", value=digits_to_str(numbers_list[-2]) if len(numbers_list) >= 2 else "")
                    actual_input = st.text_input("Actual next draw", value="")
                    use_current_preds = st.checkbox("Use current predictions", value=True)
                    pasted_preds = st.text_area("Or paste predictions (comma-separated)", value="")
                    boost_step = st.number_input("Boost step per transition", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

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
                            if best_overlap >= int(success_threshold):
                                # Apply boosts using best-matching prediction
                                if enable_boosts:
                                    best_idx = int(np.argmax(overlaps))
                                    best_pred_t = to_digits(pred_list[best_idx], 4)
                                    model_trainer.mark_success_with_boosts(seed_t, best_pred_t, enable_boosts=True, boost_step=float(boost_step))
                                    st.success("Success recorded and boosts updated.")
                                else:
                                    st.success("Success recorded (boosts disabled).")
                            else:
                                st.info("Below threshold; not considered a success.")

                st.download_button(
                    "Download predictions (CSV)",
                    pd.DataFrame({"prediction": preds_str}).to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("Please try refreshing the page to retrain the model.")

else:
    st.header("Model Performance Analysis")

    boosts = load_boosts() if enable_boosts else {}
    with st.spinner("Evaluating rolling performance..."):
        rows = model_trainer.evaluate_sequence(
            history_numbers=numbers_list,
            n_preds=30,
            mutation_strength=0.3,
            boosts=boosts,
            boost_weight=boost_weight
        )
        perf = pd.DataFrame(rows)

    if perf.empty:
        st.info("Not enough data for evaluation.")
    else:
        perf["rolling_success3"] = perf["success3"].rolling(50, min_periods=1).mean()
        perf["rolling_success4"] = perf["success4"].rolling(50, min_periods=1).mean()
        st.dataframe(perf.tail(20), use_container_width=True)

        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=perf["t"], y=perf["rolling_success3"], name="Rolling success ≥3"))
            fig.add_trace(go.Scatter(x=perf["t"], y=perf["rolling_success4"], name="Rolling success =4"))
            fig.update_layout(title="Rolling success rates", xaxis_title="t (step)", yaxis_title="Rate")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    st.subheader("Boosts status")
    current_boosts = load_boosts()
    total_pseudocounts = sum(current_boosts.values())
    st.write(f"Stored boosts: {total_pseudocounts:.1f} total across {len(current_boosts)} transitions")
    if st.button("Clear boosts"):
        save_boosts({})
        st.success("Boosts cleared.")

