"""
SentinelML - Interactive Observability & Risk Intelligence Dashboard.

Enterprise Streamlit application providing real-time transaction simulation,
SHAP explainability waterfall plots, latency SLA benchmarking, and statistical drift monitoring.
"""

import sys
from pathlib import Path

# Add project root to path for direct imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core.config import settings
from src.domain.transaction import TransactionInput
from src.ml.drift import ModelDriftDetector
from src.ml.generator import FEATURE_COLUMNS, SyntheticDataGenerator
from src.ml.inference import FraudInferenceEngine

# Page configuration
st.set_page_config(
    page_title="SentinelML | Fraud & Observability Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive polish
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #1f77b4;
    }
    .badge-critical {
        background-color: #dc3545; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold;
    }
    .badge-high {
        background-color: #fd7e14; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold;
    }
    .badge-medium {
        background-color: #ffc107; color: black; padding: 4px 10px; border-radius: 4px; font-weight: bold;
    }
    .badge-low {
        background-color: #28a745; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_engine() -> FraudInferenceEngine:
    return FraudInferenceEngine.get_instance()


@st.cache_resource(show_spinner=False)
def load_drift_detector() -> ModelDriftDetector:
    return ModelDriftDetector()


def main():
    st.title("🛡️ SentinelML: Enterprise Real-Time Fraud & Observability Engine")
    st.caption("Sub-20ms SLA Scoring • SHAP Local Attributions • Kolmogorov-Smirnov & PSI Observability")

    # Initialize ML components
    try:
        engine = load_engine()
        drift_detector = load_drift_detector()
    except Exception as e:
        st.error(f"Error initializing ML assets: {e}. Please ensure `python -m src.ml.train` was executed.")
        return

    # Sidebar system status
    with st.sidebar:
        st.subheader("System Architecture")
        st.success("● Pipeline Status: Active (LightGBM)")
        st.info("● Explainer: SHAP TreeExplainer")
        st.info("● Reference Baseline: 8,000 samples")
        st.markdown("---")
        st.subheader("SLA & Thresholds")
        st.write(f"**Target Latency:** `{settings.target_latency_ms} ms`")
        st.write(f"**Fraud Cutoff:** `{settings.fraud_probability_threshold:.2f}`")
        st.write(f"**Drift p-value (KS):** `{settings.drift_p_value_threshold:.2f}`")
        st.write(f"**Drift PSI Threshold:** `{settings.drift_psi_threshold:.2f}`")
        st.markdown("---")
        st.caption("SentinelML v1.0 • Enterprise Edition")

    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs([
        "⚡ Real-Time Transaction Simulator",
        "📊 Batch Monitoring & Latency Analytics",
        "🔬 Concept Drift & Model Health",
    ])

    # -------------------------------------------------------------
    # TAB 1: Real-Time Transaction Simulator & SHAP
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Interactive Transaction Risk Scoring & Explainability")
        st.write("Simulate an incoming transaction payload to observe real-time inference latency and local SHAP attributions.")

        # Preset scenarios
        preset = st.selectbox(
            "Load Transaction Archetype Preset:",
            [
                "Custom Parameters",
                "Legitimate: Daytime In-Store Grocery ($45, 2pm, home distance 3km)",
                "Legitimate: Evening Online Shopping ($120, 8pm, low velocity)",
                "Fraud: Midnight Credential Stuffing ($850, 3am, velocity 8, device risk 0.89)",
                "Fraud: Account Takeover & Rapid Draining ($3,400, 4am, distance 350km, device risk 0.95)",
            ],
        )

        # Default values based on preset
        defaults = {
            "amount": 65.50, "hour": 14, "distance": 4.5, "age": 36, "velocity": 1, "device": 0.15
        }
        if "Daytime In-Store" in preset:
            defaults = {"amount": 45.00, "hour": 14, "distance": 3.0, "age": 42, "velocity": 1, "device": 0.08}
        elif "Evening Online" in preset:
            defaults = {"amount": 120.00, "hour": 20, "distance": 0.5, "age": 29, "velocity": 2, "device": 0.18}
        elif "Credential Stuffing" in preset:
            defaults = {"amount": 850.00, "hour": 3, "distance": 45.0, "age": 31, "velocity": 8, "device": 0.89}
        elif "Account Takeover" in preset:
            defaults = {"amount": 3400.00, "hour": 4, "distance": 350.0, "age": 25, "velocity": 11, "device": 0.95}

        col1, col2, col3 = st.columns(3)
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=50000.0, value=float(defaults["amount"]), step=10.0)
            hour = st.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=int(defaults["hour"]))
        with col2:
            distance = st.number_input("Distance From Home (km)", min_value=0.0, max_value=2000.0, value=float(defaults["distance"]), step=5.0)
            age = st.slider("Customer Age", min_value=18, max_value=100, value=int(defaults["age"]))
        with col3:
            velocity = st.number_input("Velocity (Transactions in past 60m)", min_value=0, max_value=50, value=int(defaults["velocity"]))
            device_risk = st.slider("Device Telemetry Risk Score", min_value=0.0, max_value=1.0, value=float(defaults["device"]), step=0.01)

        tx_payload = TransactionInput(
            transaction_amount=amount,
            hour_of_day=hour,
            distance_from_home=distance,
            user_age=age,
            velocity_60m=velocity,
            device_risk_score=device_risk,
        )

        st.markdown("---")

        if st.button("🚀 Score Transaction & Generate SHAP Explanation", type="primary", use_container_width=True):
            with st.spinner("Executing real-time inference & SHAP tree explainer..."):
                explanation = engine.explain(tx_payload)
                prediction = engine.predict(tx_payload)

            # Results Section
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            with res_col1:
                st.metric("Fraud Probability", f"{prediction.risk_score * 100:.2f}%")
            with res_col2:
                decision_badge = "🔴 FRAUD DETECTED" if prediction.is_fraud else "🟢 LEGITIMATE"
                st.metric("Model Verdict", decision_badge)
            with res_col3:
                st.metric("Risk Bracket", prediction.risk_level.value)
            with res_col4:
                sla_check = "✅ Under SLA" if prediction.latency_ms < settings.target_latency_ms else "⚠️ SLA Breach"
                st.metric("Engine Latency", f"{prediction.latency_ms:.2f} ms", delta=sla_check)

            st.markdown("### 🔬 SHAP Local Feature Attribution")
            st.caption("Visualizing positive (risk-increasing) and negative (risk-mitigating) forces driving this prediction.")

            # Prepare SHAP plot data
            factors = explanation.top_risk_factors
            df_factors = pd.DataFrame([
                {
                    "Feature": f.feature,
                    "Value": f.value,
                    "SHAP Impact": f.shap_value,
                    "Direction": "Increases Fraud Risk" if f.shap_value > 0 else "Lowers Fraud Risk",
                }
                for f in factors
            ]).sort_values(by="SHAP Impact", ascending=True)

            # Plotly horizontal bar chart
            colors = ["#dc3545" if x > 0 else "#28a745" for x in df_factors["SHAP Impact"]]
            fig = go.Figure(go.Bar(
                x=df_factors["SHAP Impact"],
                y=df_factors["Feature"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in df_factors["SHAP Impact"]],
                textposition="auto",
            ))
            fig.update_layout(
                title=f"Feature Impact on Risk (Base Model Expected Log-Odds: {explanation.base_value:.2f})",
                xaxis_title="SHAP Attribution (Log-Odds Impact)",
                yaxis_title="Feature",
                height=380,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detailed attribution breakdown table
            st.markdown("#### Feature Influence Table")
            display_df = pd.DataFrame([
                {
                    "Feature Name": f.feature,
                    "Observed Input Value": f.value,
                    "SHAP Contribution": f"{f.shap_value:+.4f}",
                    "Impact Summary": "🚨 " + f.impact_direction if f.shap_value > 0 else "🛡️ " + f.impact_direction,
                }
                for f in factors
            ])
            st.dataframe(display_df, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2: Batch Monitoring & Latency Analytics
    # -------------------------------------------------------------
    with tab2:
        st.subheader("Production Traffic & Latency SLA Analytics")
        st.write("Simulate live high-concurrency traffic streams to benchmark sub-20ms p95 latency targets.")

        col_b1, col_b2 = st.columns([1, 2])
        with col_b1:
            batch_count = st.slider("Stream Volume (Transactions)", min_value=50, max_value=500, value=150, step=25)
            include_fraud = st.slider("Simulated Fraud Ratio (%)", min_value=1.0, max_value=15.0, value=4.0, step=0.5)
            run_batch = st.button("⚡ Run Streaming Batch Benchmark", type="primary")

        if run_batch:
            with st.spinner("Processing high-speed transaction pipeline..."):
                gen = SyntheticDataGenerator(seed=int(time.time()) % 1000)
                X_batch, y_true = gen.generate_baseline_dataset(
                    n_samples=batch_count, fraud_ratio=include_fraud / 100.0
                )

                latencies = []
                scores = []
                verdicts = []

                for _, row in X_batch.iterrows():
                    t_start = time.perf_counter()
                    pred = engine.predict(row.to_dict())
                    latencies.append((time.perf_counter() - t_start) * 1000)
                    scores.append(pred.risk_score)
                    verdicts.append(pred.is_fraud)

                latencies = np.array(latencies)
                p50 = np.percentile(latencies, 50)
                p95 = np.percentile(latencies, 95)
                p99 = np.percentile(latencies, 99)
                flagged_count = sum(verdicts)

            # Metric Cards
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Processed Count", f"{batch_count}")
            m2.metric("Flagged Fraud", f"{flagged_count} ({flagged_count/batch_count*100:.1f}%)")
            m3.metric("Median Latency (p50)", f"{p50:.2f} ms")
            m4.metric("p95 Latency", f"{p95:.2f} ms", delta="< 20ms SLA" if p95 < 20 else "SLA Breach")
            m5.metric("p99 Latency", f"{p99:.2f} ms")

            # Latency Distribution Plot
            fig_lat = px.histogram(
                x=latencies,
                nbins=25,
                title="End-to-End Scoring Latency Distribution (ms)",
                labels={"x": "Latency (ms)"},
                color_discrete_sequence=["#1f77b4"],
            )
            fig_lat.add_vline(x=settings.target_latency_ms, line_dash="dash", line_color="red", annotation_text="20ms SLA Boundary")
            fig_lat.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=350)
            st.plotly_chart(fig_lat, use_container_width=True)

            # Risk Score Distribution
            fig_score = px.scatter(
                x=list(range(len(scores))),
                y=scores,
                color=["Fraud" if v else "Legitimate" for v in verdicts],
                color_discrete_map={"Fraud": "#dc3545", "Legitimate": "#28a745"},
                title="Real-Time Risk Probability Stream",
                labels={"x": "Transaction Sequence", "y": "Fraud Probability"},
            )
            fig_score.add_hline(y=settings.fraud_probability_threshold, line_dash="dash", line_color="orange", annotation_text="Cutoff (0.50)")
            fig_score.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=350)
            st.plotly_chart(fig_score, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: Concept Drift & Model Health
    # -------------------------------------------------------------
    with tab3:
        st.subheader("Statistical Covariate Shift & Concept Drift Observability")
        st.write("Monitors feature distribution divergence against the baseline dataset using **Two-Sample Kolmogorov-Smirnov** and **PSI**.")

        drift_scenario = st.selectbox(
            "Select Real-World Operational Shift Scenario:",
            [
                "Stable Normal Traffic (Consistent with Baseline)",
                "Bot Velocity & Device Spoofing Shift (Credential Stuffing Attack)",
                "Economic Shock: Extreme Amount Inflation",
                "Temporal Shift: Surge in Night-Time Activity",
            ],
        )

        scenario_map = {
            "Stable Normal Traffic (Consistent with Baseline)": "normal",
            "Bot Velocity & Device Spoofing Shift (Credential Stuffing Attack)": "velocity_and_device",
            "Economic Shock: Extreme Amount Inflation": "amount_inflation",
            "Temporal Shift: Surge in Night-Time Activity": "night_owl",
        }

        if st.button("🔍 Evaluate Distribution Drift Against Baseline", type="primary"):
            with st.spinner("Calculating Kolmogorov-Smirnov statistics and PSI across features..."):
                gen = SyntheticDataGenerator(seed=123)
                chosen = scenario_map[drift_scenario]
                if chosen == "normal":
                    df_test, _ = gen.generate_baseline_dataset(n_samples=400, fraud_ratio=0.03)
                else:
                    df_test = gen.generate_drifted_batch(n_samples=400, drift_type=chosen)

                drift_report = drift_detector.evaluate_batch(df_test)

            # Banner Alert
            if drift_report.overall_drift_detected:
                st.error(
                    f"⚠️ DRIFT ALERT DETECTED: {drift_report.drifted_features_count} out of "
                    f"{len(FEATURE_COLUMNS)} features show statistically significant distribution shifts!"
                )
            else:
                st.success("✅ STABLE: All feature distributions match baseline reference dataset within statistical tolerance.")

            # Feature Report Table
            report_records = []
            for r in drift_report.feature_reports:
                report_records.append({
                    "Feature": r.feature,
                    "KS Statistic (D)": f"{r.ks_statistic:.4f}",
                    "KS p-value": f"{r.ks_p_value:.6f}",
                    "PSI": f"{r.psi:.4f}",
                    "Drift Status": "🔴 DRIFTED" if r.is_drift_detected else "🟢 STABLE",
                    "Severity": r.alert_severity,
                })
            st.dataframe(pd.DataFrame(report_records), use_container_width=True)

            # Visual distribution overlay
            st.markdown("### 📈 Baseline vs Current Batch Distribution Comparison")
            feature_to_plot = st.selectbox("Inspect Feature Distribution Overlay:", FEATURE_COLUMNS)

            base_series = drift_detector.baseline_df[feature_to_plot]
            curr_series = df_test[feature_to_plot]

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=base_series,
                histnorm="probability density",
                name="Baseline Reference (Train)",
                opacity=0.6,
                marker_color="#1f77b4",
            ))
            fig_dist.add_trace(go.Histogram(
                x=curr_series,
                histnorm="probability density",
                name="Current Production Batch",
                opacity=0.6,
                marker_color="#d62728",
            ))
            fig_dist.update_layout(
                barmode="overlay",
                title=f"Distribution Comparison for `{feature_to_plot}`",
                xaxis_title=feature_to_plot,
                yaxis_title="Density",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_dist, use_container_width=True)


if __name__ == "__main__":
    main()
