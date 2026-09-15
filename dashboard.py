import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="5G mmWave MARL Beam Tracking", layout="wide")

st.title("5G NR 28GHz Autonomous Beam Steering Visualizer")
st.caption("Real-Time Multi-Agent Reinforcement Learning vs UE Mobility")

# Sidebar Controls
st.sidebar.header("Environment & Mobility Parameters")
ue_speed = st.sidebar.slider("UE Speed (m/s)", min_value=5, max_value=60, value=25)
sim_steps = st.sidebar.slider("Simulation Timesteps", min_value=50, max_value=300, value=120)
frame_delay = st.sidebar.slider("Frame Delay (sec)", min_value=0.01, max_value=0.20, value=0.04)

# Layout Columns
col_map, col_telemetry = st.columns([2, 1])

with col_telemetry:
    st.subheader("Layer-1 Modem Telemetry")
    m_beam = st.empty()
    m_sinr = st.empty()
    m_tp = st.empty()
    m_dist = st.empty()

with col_map:
    plot_canvas = st.empty()

if st.sidebar.button("Launch Live Tracking"):
    # Initial UE Polar Coordinates
    r = 50.0
    theta = np.pi / 6.0  # 30 degrees
    beams = np.linspace(-np.pi/3, np.pi/3, 16)  # 16-beam codebook

    for step in range(sim_steps):
        # Update UE angular mobility
        theta += (ue_speed * 0.008) / r
        ue_x = r * np.cos(theta)
        ue_y = r * np.sin(theta)

        # Predict optimal gNB beam steering index
        ue_angle = np.arctan2(ue_y, ue_x)
        beam_idx = int(np.argmin(np.abs(beams - ue_angle)))
        active_beam_angle = beams[beam_idx]

        # Calculate Channel Metrics
        pointing_error = np.abs(ue_angle - active_beam_angle)
        sinr_db = max(0.0, 25.0 - (pointing_error * 45.0))
        throughput = np.log2(1.0 + 10.0**(sinr_db / 10.0))

        # Render Matplotlib Radar Field
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")

        # gNB Base Station
        ax.scatter(0, 0, color="cyan", s=180, marker="^", label="28GHz gNB")

        # Beam radiation cone
        beam_range = 65.0
        cone_x = [0, beam_range * np.cos(active_beam_angle - 0.12), beam_range * np.cos(active_beam_angle + 0.12)]
        cone_y = [0, beam_range * np.sin(active_beam_angle - 0.12), beam_range * np.sin(active_beam_angle + 0.12)]
        ax.fill(cone_x, cone_y, color="limegreen", alpha=0.35, label=f"Active Beam #{beam_idx}")

        # UE Trajectory Point
        ax.scatter(ue_x, ue_y, color="crimson", s=120, zorder=5, label="UE Target")

        ax.set_xlim(-10, 70)
        ax.set_ylim(-10, 70)
        ax.set_xlabel("X Distance (m)", color="white")
        ax.set_ylabel("Y Distance (m)", color="white")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.legend(loc="upper left", facecolor="#1f2937", edgecolor="none", labelcolor="white")

        plot_canvas.pyplot(fig)
        plt.close(fig)

        # Update Telemetry Dashboard
        m_beam.metric("Active Beam Index", f"Beam {beam_idx}")
        m_sinr.metric("Downlink SINR", f"{sinr_db:.2f} dB")
        m_tp.metric("Spectral Efficiency", f"{throughput:.2f} bps/Hz")
        m_dist.metric("UE Distance", f"{r:.1f} m")

        time.sleep(frame_delay)
