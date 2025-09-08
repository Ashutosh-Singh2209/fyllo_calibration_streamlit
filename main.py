import streamlit as st
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import plotly.express as px

load_dotenv("../.env", override=True)

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

if "plot_ids" not in st.session_state:
    st.session_state["plot_ids"] = get_plot_ids()
    st.session_state["plot_ids"].sort()

plot_ids = st.session_state["plot_ids"]

if "plot_docs" not in st.session_state:
    st.session_state["plot_docs"] = {}

if "plot_index" not in st.session_state:
    st.session_state["plot_index"] = 0

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Prev"):
        st.session_state["plot_index"] = (st.session_state["plot_index"] - 1) % len(plot_ids)
        st.session_state["plot_id_search"] = ""  
        st.rerun()
with col2:
    if st.button("Next"):
        st.session_state["plot_index"] = (st.session_state["plot_index"] + 1) % len(plot_ids)
        st.session_state["plot_id_search"] = ""   
        st.rerun()

# search_plot = st.text_input("Search plot id", key="plot_id_search")

# if search_plot:
#     filtered_plot_ids = [pid for pid in plot_ids if search_plot.lower() in pid.lower()]
# else:
#     filtered_plot_ids = plot_ids

current = plot_ids[st.session_state["plot_index"]] if plot_ids else None
# if current is not None and current not in filtered_plot_ids:
#     st.session_state["plot_id_search"] = ""
#     filtered_plot_ids = plot_ids
#     current = plot_ids[0] if plot_ids else None

start_index = 0
# if current in filtered_plot_ids:
#     start_index = filtered_plot_ids.index(current)

selected_plot_id = st.selectbox(
    "Select plot id",
    # options=filtered_plot_ids,
    options=plot_ids,
    index=start_index,
    key="plot_id_select"
)

st.session_state["plot_index"] = plot_ids.index(selected_plot_id)

if selected_plot_id not in st.session_state['plot_docs']:
    st.session_state['plot_docs'][selected_plot_id] = get_field_data(selected_plot_id)

docs = st.session_state["plot_docs"].get(selected_plot_id, [])

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

def find_calibration_in_window(a, date_times):
    pass

def find_calibration_points(a, date_times):
    b = list(a)

    def check_irr_end(i):
        return ((mean(b[i-3:i]) > 0) and (mean(b[i:i+3]) <= 0) and (b[i-1] > 0) and (b[i] <= 0))

    def check_lower_pt(i):
        return (mean(b[i-3:i]) > mean(b[i-1:i+2]) and mean(b[i+1:i+4]) > mean(b[i-1:i+2]) and mean(b[i-1:i+2]) < 0)
    # a -> array containing consecutive differences of the I1/I2 values
    # in increasing order of timestamp
    
    # hours_to_skip=4
    # avg_diff = [None]*len(a)
    # calibration_points = []
    # irrigation_detected=False
    # irrigation_peak_slope_index=None
    # irrigation_peak_slope_index_found=False
    # prev_irrigation_peak_slope_index=None
    # valley_index=None
    irr_ends = False
    irr_idx = None
    lower_pt = False
    v_i = None
    
    ans = []
    from statistics import stdev, mean
    
    for i in range(4, len(b)- 6):
        if check_irr_end(i):
            irr_ends = True
            irr_idx = i
        if irr_ends :
            for j in range(i, len(b)- 6):
                if check_lower_pt(j):
                    lower_pt = True
                    v_i = j
                # if check_irr_end(j):
                #     irr_ends = True
                #     irr_idx = j
                #     lower_pt = False
                #     break
        if irr_ends and lower_pt:
            slice_ = b[i:i+5]
            # avg_diff[i] = sum(slice_)/len(slice_)
            stdev_ = stdev(slice_)
            mean_ = mean(slice_)
            if any(x > 0 for x in b[i:i+5]):
                lower_pt = False
                continue
            if stdev_ <= 0.002 and mean_ <= 0 and mean_ > 0.5*b[v_i]:
                ans.append(date_times[i])
                irr_ends = False
                lower_pt = False
        

    # import numpy as np

    # window_size = 5
    # slope_threshold = 1e-4

    # # red_line_vals = np.array([...])

    # indices = []
    # diffs = np.diff(avg_diff)

    # for i in range(len(diffs) - window_size + 1):
    #     window = diffs[i:i + window_size]
    #     if np.std(window) <= slope_threshold:
    #         indices.append(i + window_size - 1)

    # print(indices)

    # for i in range(10, len(avg_diff)- 1, 1):
    #     for j in range(i-4, i+1, 1):


    # return 
    return ans

def exp_smooth(values, alpha=0.2):
    vals = list(values)
    n = len(vals)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([float(vals[0])], dtype=float)
    smoothed = [0.0] * n
    smoothed[0] = float(vals[0])
    for i in range(1, n):
        x = float(vals[i])
        smoothed[i] = alpha * x + (1 - alpha) * smoothed[i - 1]
    return np.array(smoothed, dtype=float)

def simple_moving_average(values, window):
    vals = list(values)
    d_deriv = False
    if vals[0] == None:
        vals = vals[1:]
        # print(f"d-deriv found")
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
        if d_deriv:
            # print(f"### running sum :{running_sum}")
            pass
        if i >= w:
            running_sum -= float(vals[i - w])
        if i >= w - 1:
            out[i] = running_sum / w
    if d_deriv:
        # print(f"*** out :{out[:20]}")
        ans = [None] + out
        # print(f"*** ans :{ans[:20]}")
        # print(f"*** ans_returned :{np.asarray(ans, dtype=float)[:20]}")
        return np.asarray(ans, dtype=float)
    return np.asarray(out, dtype=float)

def diffs_from_sorted(values):
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return np.array([], dtype=float)
    diffs = np.diff(arr)
    return diffs  # <-- keep raw diffs (not smoothed inside!)

# alpha = st.slider(
#     "SMA Window (SMA smoothing on differences)",
#     min_value=3,
#     max_value=30,
#     value=6,
#     step=1,
# )

# alpha_d_der = st.slider(
#     "SMA Window (SMA smoothing on double derivative)",
#     min_value=3,
#     max_value=30,
#     value=6,
#     step=1,
# )

# def calculate_triple_derivative(double_derivatives, window=3):
#     # remove None values from the start
#     clean_vals = [v for v in double_derivatives if v is not None]

#     if len(clean_vals) < 2:
#         return np.array([None] * len(double_derivatives), dtype=float)

#     t_deriv = [None] + list(diffs_from_sorted(clean_vals))
#     t_deriv_sma = simple_moving_average(t_deriv, window)

#     # pad to same length as input
#     while len(t_deriv_sma) < len(double_derivatives):
#         t_deriv_sma = np.append(t_deriv_sma, None)

#     return t_deriv_sma

if docs:
    df = pd.DataFrame(docs)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp", ascending=True)

    ts_diff = df["timestamp"].iloc[1:].to_list()

    if "I1" in df.columns:
        df = df.dropna(subset=["I1"])

        # print(f"getting diffs I1 {selected_plot_id}")
        I1_diff = diffs_from_sorted(df["I1"].to_numpy())

        # print(f"getting double derivative I1")
        # d_deriv_I1 = [None] + list(diffs_from_sorted(list(I1_diff)))

        # I1_exp = exp_smooth(I1_diff, alpha=alpha)
        # I1_avg = simple_moving_average(list(I1_diff), alpha)

        # print(f"getting cal points I1 {selected_plot_id}")

        # I1_cal_points = find_calibration_points(I1_avg, ts_diff)
        I1_cal_points = find_calibration_points(I1_diff, ts_diff)
        # avg_diffs1 = find_calibration_points(I1_diff, ts_diff)

        st.subheader(f"I1 values for {selected_plot_id}")

        fig_i1 = px.line(df, x="timestamp", y="I1", labels={"timestamp": "Time", "I1": "I1"})

        if selected_datetime != "(none)":
            fig_i1.add_vline(x=selected_datetime, line_width=2, line_dash="dash", line_color="green")

        for cal_datetime in I1_cal_points:
            fig_i1.add_vline(x=cal_datetime, line_width=2, line_dash="dash", line_color="red")

        st.plotly_chart(fig_i1, use_container_width=True)

        st.subheader(f"I1 differences (consecutive) for {selected_plot_id}")

        fig_i1_diff = px.line()

        fig_i1_diff.add_scatter(x=ts_diff, y=I1_diff, mode="lines", name="Original ΔI1", line=dict(color="#FFDEAD"))
        # fig_i1_diff.add_scatter(x=ts_diff, y=avg_diffs1, mode="lines", name="Avg ΔI1", line=dict(color="red"))

        # fig_i1_diff.add_scatter(x=ts_diff, y=I1_exp, mode="lines", name="Exp Smoothed", line=dict(color="red"))

        # fig_i1_diff.add_scatter(x=ts_diff, y=I1_avg, mode="lines", name="Avg Smoothed", line=dict(color="red"))

        if selected_datetime != "(none)":
            fig_i1_diff.add_vline(x=selected_datetime, line_width=2, line_dash="dash", line_color="green")

        for cal_datetime in I1_cal_points:
            fig_i1_diff.add_vline(x=cal_datetime, line_width=2, line_dash="dash", line_color="red")

        st.plotly_chart(fig_i1_diff, use_container_width=True)

        # # double derivatives
        # st.subheader(f"I1 double derivatives for {selected_plot_id}")

        # fig_i1_d_deriv = px.line()

        # # fig_i1_d_deriv.add_scatter(x=ts_diff, y=d_deriv_I1, mode="lines", name="Original d(dI1/dt)/dt", line=dict(color="#FFDEAD"))

        # d_deriv_I1_SMA = simple_moving_average(d_deriv_I1, alpha_d_der)
        # print(f"\n{d_deriv_I1[:10]}\n{d_deriv_I1_SMA[:20]}\n")

        # # fig_i1_diff.add_scatter(x=ts_diff, y=I1_exp, mode="lines", name="Exp Smoothed", line=dict(color="red"))

        # fig_i1_d_deriv.add_scatter(x=ts_diff, y=d_deriv_I1_SMA, mode="lines", name="Avg Smoothed", line=dict(color="red"))

        # if selected_datetime != "(none)":
        #     fig_i1_d_deriv.add_vline(x=selected_datetime, line_width=2, line_dash="dash", line_color="green")

        # for cal_datetime in I1_cal_points:
        #     fig_i1_d_deriv.add_vline(x=cal_datetime, line_width=2, line_dash="dash", line_color="red")

        # st.plotly_chart(fig_i1_d_deriv, use_container_width=True)
        # #double derivative logic ends

        # # triple derivatives
        # st.subheader(f"I1 triple derivatives for {selected_plot_id}")

        # fig_i1_t_deriv = px.line()

        # t_deriv_I1_SMA = calculate_triple_derivative(d_deriv_I1, window=3)

        # fig_i1_t_deriv.add_scatter(x=ts_diff, y=t_deriv_I1_SMA, mode="lines", name="Triple Derivative (SMA-3)", line=dict(color="red"))

        # if selected_datetime != "(none)":
        #     fig_i1_t_deriv.add_vline(x=selected_datetime, line_width=2, line_dash="dash", line_color="green")

        # for cal_datetime in I1_cal_points:
        #     fig_i1_t_deriv.add_vline(x=cal_datetime, line_width=2, line_dash="dash", line_color="red")

        # st.plotly_chart(fig_i1_t_deriv, use_container_width=True)

        # # triple derivatives ends

        st.subheader("Histogram of I1 differences")

        hist_i1 = px.histogram(x=I1_diff, nbins=int((max(I1_diff) - min(I1_diff))/0.0005)+1, labels={"x": "ΔI1", "y": "Count"})

        st.plotly_chart(hist_i1, use_container_width=True)

    if "I2" in df.columns:

        df = df.dropna(subset=["I2"])

        # print(f"getting diffs I2 {selected_plot_id}")
        I2_diff = diffs_from_sorted(df["I2"].to_numpy())

        # print(f"getting double derivative I2")
        # d_deriv_I2 = [None] + list(diffs_from_sorted(list(I2_diff)))

        # I2_exp = exp_smooth(I2_diff, alpha=alpha)
        # I2_avg = simple_moving_average(list(I2_diff), alpha)

        # print(f"getting cal points I2 {selected_plot_id}")

        # I2_cal_points = find_calibration_points(I2_avg, ts_diff)
        I2_cal_points = find_calibration_points(I2_diff, ts_diff)
        # avg_diffs2 = find_calibration_points(I2_diff, ts_diff)

        st.subheader(f"I2 values for {selected_plot_id}")

        fig_i2 = px.line(df, x="timestamp", y="I2", labels={"timestamp": "Time", "I2": "I2"})

        if selected_datetime != "(none)":
            fig_i2.add_vline(x=selected_datetime, line_width=2, line_dash="dash", line_color="green")

        for cal_datetime in I2_cal_points:
            fig_i2.add_vline(x=cal_datetime, line_width=2, line_dash="dash", line_color="red")

        st.plotly_chart(fig_i2, use_container_width=True)

        st.subheader(f"I2 differences (consecutive) for {selected_plot_id}")

        fig_i2_diff = px.line()

        fig_i2_diff.add_scatter(x=ts_diff, y=I2_diff, mode="lines", name="Original ΔI2", line=dict(color="#FFDEAD"))
        # fig_i2_diff.add_scatter(x=ts_diff, y=avg_diffs2, mode="lines", name="Avg ΔI2", line=dict(color="red"))
        # fig_i2_diff.add_scatter(x=ts_diff, y=I2_exp, mode="lines", name="Exp Smoothed", line=dict(color="red"))

        # fig_i2_diff.add_scatter(x=ts_diff, y=I2_avg, mode="lines", name="Avg Smoothed", line=dict(color="red"))

        if selected_datetime != "(none)":
            fig_i2_diff.add_vline(x=selected_datetime, line_width=2, line_dash="dash", line_color="green")

        for cal_datetime in I2_cal_points:
            fig_i2_diff.add_vline(x=cal_datetime, line_width=2, line_dash="dash", line_color="red")

        st.plotly_chart(fig_i2_diff, use_container_width=True)

        # # double derivatives
        # st.subheader(f"I2 double derivatives for {selected_plot_id}")

        # fig_i2_d_deriv = px.line()

        # # fig_i2_d_deriv.add_scatter(x=ts_diff, y=d_deriv_I2, mode="lines", name="Original d(dI2/dt)/dt", line=dict(color="#FFDEAD"))

        # d_deriv_I2_SMA = simple_moving_average(d_deriv_I2, alpha_d_der)
        # print(f"\n{d_deriv_I2[:10]}\n{d_deriv_I2_SMA[:20]}\n")

        # # fig_i2_diff.add_scatter(x=ts_diff, y=I2_exp, mode="lines", name="Exp Smoothed", line=dict(color="red"))

        # fig_i2_d_deriv.add_scatter(x=ts_diff, y=d_deriv_I2_SMA, mode="lines", name="Avg Smoothed", line=dict(color="red"))

        # if selected_datetime != "(none)":
        #     fig_i2_d_deriv.add_vline(x=selected_datetime, line_width=2, line_dash="dash", line_color="green")

        # for cal_datetime in I2_cal_points:
        #     fig_i2_d_deriv.add_vline(x=cal_datetime, line_width=2, line_dash="dash", line_color="red")

        # st.plotly_chart(fig_i2_d_deriv, use_container_width=True)
        # #double derivative logic ends

        # # triple derivatives
        # st.subheader(f"I2 triple derivatives for {selected_plot_id}")

        # fig_i2_t_deriv = px.line()

        # t_deriv_I2_SMA = calculate_triple_derivative(d_deriv_I2, window=3)

        # fig_i2_t_deriv.add_scatter(x=ts_diff, y=t_deriv_I2_SMA, mode="lines", name="Triple Derivative (SMA-3)", line=dict(color="red"))

        # if selected_datetime != "(none)":
        #     fig_i2_t_deriv.add_vline(x=selected_datetime, line_width=2, line_dash="dash", line_color="green")

        # for cal_datetime in I2_cal_points:
        #     fig_i2_t_deriv.add_vline(x=cal_datetime, line_width=2, line_dash="dash", line_color="red")

        # st.plotly_chart(fig_i2_t_deriv, use_container_width=True)

        # # tripple derivative logic ends

        st.subheader("Histogram of I2 differences")

        hist_i2 = px.histogram(x=I2_diff, nbins=int((max(I2_diff) - min(I2_diff))//0.0005)+1, labels={"x": "ΔI2", "y": "Count"})

        st.plotly_chart(hist_i2, use_container_width=True)

else:
    st.info("No data found for the selected plot id.")
