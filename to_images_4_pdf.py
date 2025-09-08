import os
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv("../.env", override=True)

mongo_uri = os.getenv("FYLLO_MONGO_URI")
client = MongoClient(mongo_uri)
db = client["database"]
device_collection = db["device"]
field_data_collection = db["FinalFieldData"]

def get_plot_ids():
    q = {
        "deviceType": "NERO_INFINITY_UNIT",
        "installationDate": {"$gte": datetime(2025, 7, 25)},
        "isAssigned": True
    }
    return [d["plotId"] for d in device_collection.find(q, {"plotId": 1, "_id": 0}) if "plotId" in d]

def get_field_data(plot_id):
    q = {
        "plotId": plot_id,
        "timestamp": {"$gte": datetime(2025, 7, 25)}
    }
    p = {
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
    docs = list(field_data_collection.find(q, p))
    return docs

def diffs_from_sorted(values):
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return np.array([], dtype=float)
    return np.diff(arr)

def find_calibration_points(a, date_times):
    # a -> array containing consecutive differences of the I1/I2 values
    # in increasing order of timestamp
    hours_to_skip=5
    calibration_points = []
    global alpha
    irrigation_detected=False
    irrigation_peak_slope_index=None
    irrigation_peak_slope_index_found=False
    prev_irrigation_peak_slope_index=None
    valley_index=None
    for i in range(0, len(a), 1):
        # break
        if a[i] > 0:
            # print(f"cond 1")
            irrigation_detected=True
        if a[i-1] > 0 and (a[i-1]>0.01):
            # print(f"cond 2")
            irrigation_peak_slope_index_found=True
            if irrigation_peak_slope_index:
                prev_irrigation_peak_slope_index=irrigation_peak_slope_index
            irrigation_peak_slope_index=i-1
        if a[i] <= 0 and a[i-1] < 0 and (a[i-1]-a[i])<0:
            # print(f"cond 3")
            if irrigation_detected and irrigation_peak_slope_index_found:
                # print(f"cond 3.1")
                # diff_hours = (date_times[i-1] - date_times[irrigation_peak_slope_index]).total_seconds() / 3600
                # if diff_hours > 0:
                #     print(f"cond 3.1.1")
                valley_point = a[i-1]
                while (a[i] < -0.005) and (a[i]<0):
                    # print(f"cond 3.1.1")
                    i+=1
                    if i >= len(a):
                        return calibration_points
                # calibration_points.append(date_times[i-1+hours_to_skip])
                calibration_points.append(date_times[i])
            # print(calibration_points[-10:])
                irrigation_detected=False
                irrigation_peak_slope_index_found=False

    return calibration_points

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def to_dt_list(ts):
    return pd.to_datetime(ts).tolist()

from test2 import find_calibration_points2

def plot_and_save(plot_id, df, out_dir):
    fig = plt.figure(figsize=(10, 16), constrained_layout=True)
    gs = fig.add_gridspec(4, 1)

    ax_i1 = fig.add_subplot(gs[0, 0])
    ax_i1d = fig.add_subplot(gs[1, 0])
    ax_i2 = fig.add_subplot(gs[2, 0])
    ax_i2d = fig.add_subplot(gs[3, 0])

    dfi1 = df.dropna(subset=["I1"]).copy() if "I1" in df.columns else pd.DataFrame()
    dfi2 = df.dropna(subset=["I2"]).copy() if "I2" in df.columns else pd.DataFrame()

    ts_i1 = dfi1["timestamp"].sort_values().tolist() if not dfi1.empty else []
    ts_i2 = dfi2["timestamp"].sort_values().tolist() if not dfi2.empty else []

    i1_vals = dfi1.sort_values("timestamp")["I1"].to_numpy() if not dfi1.empty else np.array([])
    i2_vals = dfi2.sort_values("timestamp")["I2"].to_numpy() if not dfi2.empty else np.array([])

    ts_i1_diff = dfi1.sort_values("timestamp")["timestamp"].iloc[1:].tolist() if len(ts_i1) >= 2 else []
    ts_i2_diff = dfi2.sort_values("timestamp")["timestamp"].iloc[1:].tolist() if len(ts_i2) >= 2 else []

    i1_diff = diffs_from_sorted(i1_vals) if i1_vals.size >= 2 else np.array([])
    i2_diff = diffs_from_sorted(i2_vals) if i2_vals.size >= 2 else np.array([])

    i1_cal = find_calibration_points2(i1_diff, ts_i1_diff) if len(ts_i1_diff) == len(i1_diff) and len(i1_diff) > 0 else []
    i2_cal = find_calibration_points2(i2_diff, ts_i2_diff) if len(ts_i2_diff) == len(i2_diff) and len(i2_diff) > 0 else []

    # if not dfi1.empty:
    #     sns.lineplot(x="timestamp", y="I1", data=dfi1.sort_values("timestamp"), ax=ax_i1, color="#1f77b4")
    #     ax_i1.set_title(f"I1 Time Series: {plot_id}")
    #     ax_i1.set_xlabel("Time")
    #     ax_i1.set_ylabel("I1")
    #     for dt in i1_cal:
    #         ax_i1.axvline(dt, color="red", linestyle="--", linewidth=1.5)

    # if len(i1_diff) > 0 and len(ts_i1_diff) > 0:
    #     ax_i1d.plot(ts_i1_diff, i1_diff, color="#ff7f0e")
    #     ax_i1d.set_title(f"I1 Differences: {plot_id}")
    #     ax_i1d.set_xlabel("Time")
    #     ax_i1d.set_ylabel("ΔI1")
    #     for dt in i1_cal:
    #         ax_i1d.axvline(dt, color="red", linestyle="--", linewidth=1.5)

    # if not dfi2.empty:
    #     sns.lineplot(x="timestamp", y="I2", data=dfi2.sort_values("timestamp"), ax=ax_i2, color="#2ca02c")
    #     ax_i2.set_title(f"I2 Time Series: {plot_id}")
    #     ax_i2.set_xlabel("Time")
    #     ax_i2.set_ylabel("I2")
    #     for dt in i2_cal:
    #         ax_i2.axvline(dt, color="red", linestyle="--", linewidth=1.5)

    # if len(i2_diff) > 0 and len(ts_i2_diff) > 0:
    #     ax_i2d.plot(ts_i2_diff, i2_diff, color="#9467bd")
    #     ax_i2d.set_title(f"I2 Differences: {plot_id}")
    #     ax_i2d.set_xlabel("Time")
    #     ax_i2d.set_ylabel("ΔI2")
    #     for dt in i2_cal:
    #         ax_i2d.axvline(dt, color="red", linestyle="--", linewidth=1.5)

    # fig.suptitle(f"Calibration Visualization for Plot {plot_id}", fontsize=16)

    # ensure_dir(out_dir)
    # safe_id = str(plot_id).replace("/", "_").replace("\\", "_").replace(" ", "_")
    # fpath = os.path.join(out_dir, f"{safe_id}.png")
    # fig.savefig(fpath, dpi=150)
    # plt.close(fig)

    # return fpath

def main():
    from tqdm import tqdm
    out_dir = "calibration_plots2"
    plot_ids = sorted(list(set(get_plot_ids())))
    for i, pid in enumerate(tqdm(plot_ids)):
        # if i < 133:
        #     continue
        docs = get_field_data(pid)
        if not docs:
            continue
        df = pd.DataFrame(docs)
        if "timestamp" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        if "I1" not in df.columns and "I2" not in df.columns:
            continue
        plot_and_save(pid, df, out_dir)

if __name__ == "__main__":
    main()
