import streamlit as st
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import plotly.express as px
from collections import deque

load_dotenv("../.env", override=True)
# load_dotenv()

class LRUCache:
    def __init__(self, max_size=50):
        self.cache = {}
        self.order = deque()
        self.max_size = max_size
    
    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.max_size:
            oldest_key = self.order.popleft()
            del self.cache[oldest_key]
        
        self.cache[key] = value
        self.order.append(key)
    
    def __contains__(self, key):
        return key in self.cache

if "mongo_client" not in st.session_state:
    st.session_state["mongo_client"] = MongoClient(os.getenv('FYLLO_MONGO_URI'))
    db = st.session_state["mongo_client"]["database"]
    st.session_state["device_collection"] = db["device"]
    st.session_state["field_data_collection"] = db["FinalFieldData"]

if "plot_docs_cache" not in st.session_state:
    st.session_state["plot_docs_cache"] = LRUCache(50)

device_collection = st.session_state["device_collection"]
field_data_collection = st.session_state["field_data_collection"]

def get_plot_ids():
    filter_query = {
        "deviceType": "NERO_INFINITY_UNIT",
        "installationDate": {"$gte": datetime(2025, 7, 25)},
        "isAssigned": True
    }
    return [
        doc["plotId"] for doc in device_collection.find(filter_query, {"plotId": 1, "_id": 0})
    ]

def get_field_data(plot_id):
    filter_query = {
        "plotId": plot_id,
        "timestamp": {"$gte": datetime(2025, 7, 25)}
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

st.set_page_config(page_title="Plot I1 & I2 Visualizer", layout="wide")

plot_numbers = [
    "plotno1mamajumbo273",
    "plotno1sudhakar444",
    "plotno3crimson883",
    "plotno1335",
    "plotno1sudhakar162727",
    "plotno2sudhakar118",
    "plotno15bsudhakar",
    "plotno1pomegranate601",
    "plotno4sudhakarh",
    "plotno3ssn894",
    "plotno1thompson197",
    "plotno5thomspon",
    "plotno1jumbo736",
    "plotno2sudhakar381",
    "plotno3thomspon",
    "plotno2sonaka611",
    "plotnomamajumbomanoj",
    "plotnothompsonvishal",
    "plotno2rk356",
    "plotno237",
    "plotno2ssn302"
]

if "plot_ids" not in st.session_state:
    # st.session_state["plot_ids"] = get_plot_ids()
    st.session_state["plot_ids"] = plot_numbers
    st.session_state["plot_ids"].sort()

plot_ids = st.session_state["plot_ids"]

if "plot_index" not in st.session_state:
    st.session_state["plot_index"] = 0

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Prev"):
        st.session_state["plot_index"] = (st.session_state["plot_index"] - 1) % len(plot_ids)
        st.rerun()

with col2:
    if st.button("Next"):
        st.session_state["plot_index"] = (st.session_state["plot_index"] + 1) % len(plot_ids)
        st.rerun()

current_plot_id = plot_ids[st.session_state["plot_index"]] if plot_ids else None

selected_plot_id = st.selectbox(
    "Select plot id",
    options=plot_ids,
    index=st.session_state["plot_index"],
    key="plot_id_select"
)

if selected_plot_id and plot_ids:
    new_index = plot_ids.index(selected_plot_id)
    if new_index != st.session_state["plot_index"]:
        st.session_state["plot_index"] = new_index

cached_data = st.session_state["plot_docs_cache"].get(selected_plot_id)
if cached_data is None:
    docs = get_field_data(selected_plot_id)
    st.session_state["plot_docs_cache"].put(selected_plot_id, docs)
else:
    docs = cached_data

available_datetimes = []
if docs:
    _df_dates = pd.DataFrame(docs)
    _df_dates["timestamp"] = pd.to_datetime(_df_dates["timestamp"])
    available_datetimes = sorted(_df_dates["timestamp"].unique().tolist())

selected_datetime = st.selectbox(
    "Mark a datetime (optional)",
    options=["(none)"] + [dt for dt in available_datetimes],
    index=0,
    format_func=lambda v: v if v == "(none)" else v.strftime("%Y-%m-%d %H:%M:%S"),
    key="selected_datetime_mark"
)

selected_index = None
if selected_datetime != "(none)":
    selected_index = available_datetimes.index(selected_datetime)
    st.write(f"**Selected datetime index:** {selected_index}")
else:
    st.write("**Selected datetime index:** None")

from test2 import find_calibration_points2, find_calibration_points4

def find_calibration_points(a, date_times):
    calibration_points = []
    irrigation_detected=False
    irrigation_peak_slope_index=None
    irrigation_peak_slope_index_found=False
    prev_irrigation_peak_slope_index=None
    valley_index=None
    for i in range(1, len(a) - 1, 1):
        if a[i] > 0:
            irrigation_detected=True
        if a[i-1] > 0 and (a[i-1]>0.01):
            irrigation_peak_slope_index_found=True
            if irrigation_peak_slope_index:
                prev_irrigation_peak_slope_index=irrigation_peak_slope_index
            irrigation_peak_slope_index=i-1
        if a[i] <= 0 and a[i-1] < 0 and (a[i-1]-a[i])<0 and a[i-1] < -0.0018:
            if irrigation_detected and irrigation_peak_slope_index_found:
                valley_point = a[i-1]
                while (a[i-1] < -0.005) and (a[i]<0):
                    i+=1
                    if i >= len(a):
                        return calibration_points
                    
                calibration_points.append(date_times[i-1])
                irrigation_detected=False
                irrigation_peak_slope_index_found=False
    return calibration_points

def simple_moving_average(values, window):
    vals = list(values)
    d_deriv = False
    if vals[0] == None:
        vals = vals[1:]
        d_deriv=True
    n = len(vals)
    w = int(window)
    if w <= 0:
        raise ValueError("window must be a positive integer")
    if n == 0:
        return np.array([], dtype=float)
    if w == 1:
        return np.asarray([float(v) for v in vals], dtype=float)
    out = [float("nan")] * n
    running_sum = 0.0
    for i in range(n):
        running_sum += float(vals[i])
        if i >= w:
            running_sum -= float(vals[i - w])
        if i >= w - 1:
            out[i] = running_sum / w
    if d_deriv:
        ans = [None] + out
        return np.asarray(ans, dtype=float)
    return np.asarray(out, dtype=float)

def diffs_from_sorted(values):
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return np.array([], dtype=float)
    diffs = np.diff(arr)
    return diffs

if docs:
    df = pd.DataFrame(docs)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp", ascending=True)

    ts_diff = df["timestamp"].iloc[1:].to_list()

    if "I1" in df.columns:
        df = df.dropna(subset=["I1"])

        I1_diff = diffs_from_sorted(df["I1"].to_numpy())

        d_deriv_I1 = [None] + list(diffs_from_sorted(list(I1_diff)))

        I1_cal_points, avg_cal_val1 = find_calibration_points4(list(df["I1"].to_numpy()[1:]), ts_diff)
        # I1_cal_points = find_calibration_points(I1_diff, ts_diff)

        st.subheader(f"I1 values for {selected_plot_id}")
        st.write(f"avg calibration value : {avg_cal_val1}")

        fig_i1 = px.line(df, x="timestamp", y="I1", labels={"timestamp": "Time", "I1": "I1"})

        if selected_datetime != "(none)":
            fig_i1.add_vline(x=selected_datetime, line_width=2, line_dash="dash", line_color="green")

        for cal_datetime in I1_cal_points:
            fig_i1.add_vline(x=cal_datetime, line_width=2, line_dash="dash", line_color="red")

        st.plotly_chart(fig_i1, use_container_width=True)

        st.subheader(f"I1 differences (consecutive) for {selected_plot_id}")

        fig_i1_diff = px.line()

        fig_i1_diff.add_scatter(x=ts_diff, y=I1_diff, mode="lines", name="Original ΔI1", line=dict(color="#FFDEAD"))

        if selected_datetime != "(none)":
            fig_i1_diff.add_vline(x=selected_datetime, line_width=2, line_dash="dash", line_color="green")

        for cal_datetime in I1_cal_points:
            fig_i1_diff.add_vline(x=cal_datetime, line_width=2, line_dash="dash", line_color="red")

        st.plotly_chart(fig_i1_diff, use_container_width=True)

        st.subheader("Histogram of I1 differences")

        hist_i1 = px.histogram(x=I1_diff, nbins=50, labels={"x": "ΔI1", "y": "Count"})

        st.plotly_chart(hist_i1, use_container_width=True)

    if "I2" in df.columns:

        df = df.dropna(subset=["I2"])

        I2_diff = diffs_from_sorted(df["I2"].to_numpy())

        I2_cal_points, avg_cal_val2 = find_calibration_points4(list(df["I2"].to_numpy()[1:]), ts_diff)
        # I2_cal_points = find_calibration_points(I2_diff, ts_diff)

        st.subheader(f"I2 values for {selected_plot_id}")
        st.write(f"avg calibration value : {avg_cal_val2}")

        fig_i2 = px.line(df, x="timestamp", y="I2", labels={"timestamp": "Time", "I2": "I2"})

        if selected_datetime != "(none)":
            fig_i2.add_vline(x=selected_datetime, line_width=2, line_dash="dash", line_color="green")

        for cal_datetime in I2_cal_points:
            fig_i2.add_vline(x=cal_datetime, line_width=2, line_dash="dash", line_color="red")

        st.plotly_chart(fig_i2, use_container_width=True)

        st.subheader(f"I2 differences (consecutive) for {selected_plot_id}")

        fig_i2_diff = px.line()

        fig_i2_diff.add_scatter(x=ts_diff, y=I2_diff, mode="lines", name="Original ΔI2", line=dict(color="#FFDEAD"))

        if selected_datetime != "(none)":
            fig_i2_diff.add_vline(x=selected_datetime, line_width=2, line_dash="dash", line_color="green")

        for cal_datetime in I2_cal_points:
            fig_i2_diff.add_vline(x=cal_datetime, line_width=2, line_dash="dash", line_color="red")

        st.plotly_chart(fig_i2_diff, use_container_width=True)

        st.subheader("Histogram of I2 differences")

        hist_i2 = px.histogram(x=I2_diff, nbins=50, labels={"x": "ΔI2", "y": "Count"})

        st.plotly_chart(hist_i2, use_container_width=True)

else:
    st.info("No data found for the selected plot id.")
