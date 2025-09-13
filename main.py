import streamlit as st
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import numpy as np
import plotly.express as px
from utils import find_calibration_points, get_calibration_value, percentile_min_max_scale
from functools import lru_cache

load_dotenv(find_dotenv(), override=True)

if "window_size" not in st.session_state:
    st.session_state['window_size'] = 5
if "threshold" not in st.session_state:
    st.session_state['threshold'] = 2.0
if "threshold_2" not in st.session_state:
    st.session_state['threshold_2'] = 2.0
if "prev_slope" not in st.session_state:
    st.session_state['prev_slope'] = -0.001
if "curr_slope" not in st.session_state:
    st.session_state['curr_slope'] = 0

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.number_input("Enter window size:", key="window_size", step=1, min_value=None)
with col2:
    st.number_input("Enter threshold:", key="threshold", step=0.1, format="%.4f", min_value=None)
with col3:
    st.number_input("Enter threshold_2:", key="threshold_2", step=0.1, format="%.4f", min_value=None)
with col4:
    st.number_input("Enter prev_slope:", key="prev_slope", step=0.0001, format="%.5f", min_value=None)
with col5:
    st.number_input("Enter curr_slope:", key="curr_slope", step=0.0001, format="%.5f", min_value=None)

if "mongo_client" not in st.session_state:
    st.session_state["mongo_client"] = MongoClient(os.getenv('FYLLO_MONGO_URI'))
    db = st.session_state["mongo_client"]["database"]
    st.session_state["device_collection"] = db["device"]
    st.session_state["field_data_collection"] = db["FinalFieldData"]

device_collection = st.session_state["device_collection"]
field_data_collection = st.session_state["field_data_collection"]

def get_plot_ids():
    filter_query = {
        "deviceType": "NERO_INFINITY_UNIT",
        "installationDate": {"$gte": datetime(2025, 7, 10)},
        "isAssigned": True
    }
    return [
        doc["plotId"] for doc in device_collection.find(filter_query, {"plotId": 1, "_id": 0})
    ]

def get_field_data(plot_id):
    filter_query = {
        "plotId": plot_id,
        "timestamp": {"$gte": datetime(2025, 7, 10)}
    }
    projection = {
        "_id": 1,
        "deviceId": 1,
        "plotId": 1,
        "farmUserId": 1,
        "timestamp": 1,
        "moisture1": 1,
        "moisture2": 1,
        "I1": 1,
        "I2": 1
    }
    docs = list(field_data_collection.find(filter_query, projection))
    return docs

if "cached_get_field_data" not in st.session_state:
    st.session_state["cached_get_field_data"] = lru_cache(maxsize=300)(get_field_data)

st.set_page_config(page_title="Plot I1 & I2 Visualizer", layout="wide")

if "plot_ids" not in st.session_state:
    st.session_state["plot_ids"] = get_plot_ids()
    st.session_state["plot_ids"].sort()

plot_ids = st.session_state["plot_ids"]
col1, col2 = st.columns(2)
with col1:
    st.caption(f"Number of plots: {len(plot_ids)}")
with col2:
    if "plot_id_select" in st.session_state:
        st.caption(f"Current Index: {plot_ids.index(st.session_state['plot_id_select'])}")
    else:
        st.caption(f"Current Index: {0}")

if "plot_index" not in st.session_state:
    st.session_state["plot_index"] = 0

col1, col2, col3, col4 = st.columns([1,1,2,2])

with col1:
    if st.button("Prev"):
        st.session_state["plot_index"] = (st.session_state["plot_index"] - 1) % len(plot_ids)
        st.session_state["selected_plot_id"] = plot_ids[st.session_state["plot_index"]]
        st.session_state["plot_id_select"] = plot_ids[st.session_state["plot_index"]]
        st.session_state["plot_id_search"] = ""
        st.rerun()

with col2:
    if st.button("Next"):
        st.session_state["plot_index"] = (st.session_state["plot_index"] + 1) % len(plot_ids)
        st.session_state["selected_plot_id"] = plot_ids[st.session_state["plot_index"]]
        st.session_state["plot_id_select"] = plot_ids[st.session_state["plot_index"]]
        st.session_state["plot_id_search"] = ""
        st.rerun()


current = plot_ids[st.session_state["plot_index"]] if plot_ids else None

start_index = 0
with col3:
    selected_plot_id = st.selectbox(
        "Select plot id",
        options=plot_ids,
        index=start_index,
        key="plot_id_select"
    )

st.session_state["plot_index"] = plot_ids.index(selected_plot_id)

docs = st.session_state["cached_get_field_data"](selected_plot_id)

available_datetimes = []
if docs:
    _df_dates = pd.DataFrame(docs)
    _df_dates["timestamp"] = pd.to_datetime(_df_dates["timestamp"])
    available_datetimes = sorted(_df_dates["timestamp"].unique().tolist())

with col4:
    selected_datetime = st.selectbox(
        "Mark a datetime (optional)",
        options=["(none)"] + [dt for dt in available_datetimes],
        index=0,
        format_func=lambda v: v if v == "(none)" else v.strftime("%Y-%m-%d %H:%M:%S"),
        key="selected_datetime_mark"
    )

if "show_I1" not in st.session_state:
    st.session_state["show_I1"] = True
if "show_values" not in st.session_state:
    st.session_state["show_values"] = True
if "show_slopes" not in st.session_state:
    st.session_state["show_slopes"] = True
if "show_histograms" not in st.session_state:
    st.session_state["show_histograms"] = True
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("Show I1"):
        st.session_state["show_I1"] = not st.session_state["show_I1"]
with c2:
    if st.button("Show Values"):
        st.session_state["show_values"] = not st.session_state["show_values"]
with c3:
    if st.button("Show Slopes"):
        st.session_state["show_slopes"] = not st.session_state["show_slopes"]
with c4:
    if st.button("Show Histograms"):
        st.session_state["show_histograms"] = not st.session_state["show_histograms"]

def diffs_from_sorted(values):
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return np.array([], dtype=float)
    diffs = np.diff(arr)
    return diffs  

window, threshold, threshold_2, prev_slope, curr_slope = st.session_state["window_size"], st.session_state["threshold"], st.session_state["threshold_2"], st.session_state["prev_slope"], st.session_state["curr_slope"]
import numpy as np
from make_histogram import slope_histogram

if docs:
    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", ascending=True)
    # df = df.tail(10*24)
    ts_diff = df["timestamp"].iloc[1:].to_list()

    hist_col1, hist_col2 = st.columns(2)

    if "I1" in df.columns and st.session_state["show_I1"]:
        
        from utils import time_to_90, plot_t90_histogram
        df = df.dropna(subset=["I1"])
        # df["I1"] = percentile_min_max_scale(df["I1"].to_numpy())
        I1_diff = diffs_from_sorted(df["I1"].to_numpy())
        I1_cal_points, I1_cal_indices = find_calibration_points(df['I1'].to_list(), df["timestamp"].to_list(),\
        window, threshold, threshold_2, prev_slope, curr_slope)
        I1_cal_value = get_calibration_value(df["I1"].to_list(), I1_cal_indices)
        # t90_df = time_to_90(df, col="I1")
        # st.plotly_chart(plot_t90_histogram(t90_df), use_container_width=True)
        if st.session_state["show_values"]:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader(f"I1 values for {selected_plot_id}")
            with c2:
                st.caption(f"I1 calibration value: {I1_cal_value}")

            fig_i1 = px.line(df, x="timestamp", y="I1", labels={"timestamp": "Time", "I1": "I1"})
            fig_i1.update_yaxes(title_text=None)

            if selected_datetime != "(none)":
                fig_i1.add_vline(x=selected_datetime, line_width=2, line_dash="dash", line_color="green")
            # fig_i1.add_hline(y=1.005, line_width=2, line_dash="dash", line_color="green")
            for cal_datetime in I1_cal_points:
                fig_i1.add_vline(x=cal_datetime, line_width=2, line_dash="dash", line_color="red")

            st.plotly_chart(fig_i1, use_container_width=True)

        # st.subheader(f"I1 distribution {selected_plot_id}")

        # fig_i_hist = histogram(df, "I1")

        # st.plotly_chart(fig_i_hist, use_container_width=True)

        if st.session_state["show_slopes"]:
            st.subheader(f"I1 differences (consecutive) for {selected_plot_id}")
            fig_i1_diff = px.line()
            fig_i1_diff.add_scatter(x=df["timestamp"].to_list(), y=[None]+list(I1_diff), mode="lines", name="Original ΔI1", line=dict(color="#FFDEAD"))

            if selected_datetime != "(none)":
                fig_i1_diff.add_vline(x=selected_datetime, line_width=2, line_dash="dash", line_color="green")

            # fig_i1_diff.add_hline(y=np.percentile(I1_diff[I1_diff>0], 70), line_width=2, line_dash="dash", line_color="green")
            for cal_datetime in I1_cal_points:
                fig_i1_diff.add_vline(x=cal_datetime, line_width=2, line_dash="dash", line_color="red")

            # fig_i1_diff.add_hline(y=0.01, line_width=2, line_dash="dash", line_color="red")

            st.plotly_chart(fig_i1_diff, use_container_width=True)

        if st.session_state["show_histograms"]:
            with hist_col1:
                st.subheader("Histogram of I1 differences")

                if len(I1_diff) > 0:
                    hist_i1 = slope_histogram(I1_diff)
                    st.plotly_chart(hist_i1, use_container_width=True)
                else:
                    st.warning("No data available for I1 differences.")

    if "I2" in df.columns:
        # from utils import time_to_90, plot_t90_histogram
        df = df.dropna(subset=["I2"])
        # df["I2"] = percentile_min_max_scale(df["I2"].to_numpy())
        I2_diff = diffs_from_sorted(df["I2"].to_numpy())
        I2_cal_points, I2_cal_indices = find_calibration_points(df["I2"].to_list(), df["timestamp"].to_list(),\
        window, threshold, threshold_2, prev_slope, curr_slope)
        I2_cal_value = get_calibration_value(df["I2"].to_list(), I2_cal_indices)
        # t90_df = time_to_90(df, col="I2")
        # st.plotly_chart(plot_t90_histogram(t90_df), use_container_width=True)

        if st.session_state["show_values"]:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader(f"I2 values for {selected_plot_id}")
            with c2:
                st.caption(f"I2 calibration value: {I2_cal_value}")

            fig_i2 = px.line(df, x="timestamp", y="I2", labels={"timestamp": "Time", "I2": "I2"})
            fig_i2.update_yaxes(title_text=None)

            if selected_datetime != "(none)":
                fig_i2.add_vline(x=selected_datetime, line_width=2, line_dash="dash", line_color="green")

            for cal_datetime in I2_cal_points:
                fig_i2.add_vline(x=cal_datetime, line_width=2, line_dash="dash", line_color="red")

            st.plotly_chart(fig_i2, use_container_width=True)
        
        if st.session_state["show_slopes"]:

            st.subheader(f"I2 differences (consecutive) for {selected_plot_id}")
            fig_i2_diff = px.line()
            fig_i2_diff.add_scatter(x=df["timestamp"].to_list(), y=[None]+list(I2_diff), mode="lines", name="Original ΔI2", line=dict(color="#FFDEAD"))

            if selected_datetime != "(none)":
                fig_i2_diff.add_vline(x=selected_datetime, line_width=2, line_dash="dash", line_color="green")
            
            # fig_i2_diff.add_hline(y=np.percentile(I2_diff[I2_diff>0], 70), line_width=2, line_dash="dash", line_color="green")
            for cal_datetime in I2_cal_points:
                fig_i2_diff.add_vline(x=cal_datetime, line_width=2, line_dash="dash", line_color="red")
            # fig_i2_diff.add_hline(y=0.01, line_width=2, line_dash="dash", line_color="red")

            st.plotly_chart(fig_i2_diff, use_container_width=True)

        if st.session_state["show_histograms"]:
            with hist_col2:
                st.subheader("Histogram of I2 differences")

                if len(I2_diff) > 0:
                    hist_i2 = slope_histogram(I2_diff)
                    st.plotly_chart(hist_i2, use_container_width=True)
                else:
                    st.warning("No data available to plot histogram of I2 differences.")


else:
    st.info("No data found for the selected plot id.")
