from statistics import stdev, mean
import numpy as np

def find_calibration_points_(a, date_times):
    b = list(a)

    def check_irr_end(i):
        return ((mean(b[i-3:i]) > 0) and (mean(b[i:i+3]) <= 0) and (b[i-1] > 0) and (b[i] <= 0))

    def check_lower_pt(i):
        cond1 = (mean(b[i-2:i]) > b[i] and mean(b[i+1:i+4]) > b[i] and b[i] < 0)
        cond2 = np.all(np.array(b[i:i+4]) <= 0)
        cond3 = irr_ends and irr_detected and i > irr_idx and lower_pt == False
        return cond1 and cond2

    def check_break_point(i):
        cond1 = b[i] <= 0
        cond2 = b[i] >= 0.6*b[v_i]
        return cond1 and cond2

    def check_if_valley_point(i):
        cond0 = cal_indices and i > cal_indices[-1]-1
        cond1 = irr_idx is not None and v_i is not None
        cond2 = irr_ends == False and lower_pt == False and irr_detected == False
        # cond3 = check_lower_pt(i)
        cond4 = (b[i] < 0.55*b[v_i]) if cond1 else True
        return cond0 and cond1 and cond2 and cond4 #and cond4

    irr_ends = False
    irr_idx = None
    lower_pt = False
    v_i = None
    irr_detected = False
    irr_point = None

    ans = []
    cal_indices = []

    for i in range(4, len(b)-6):
        if b[i] > 0.01:
            irr_detected = True
            irr_point = i
        if irr_detected and check_irr_end(i) and i > irr_point and irr_ends == False:
            irr_ends = True
            irr_idx = i
        if irr_ends and irr_detected and check_lower_pt(i) and i > irr_idx and lower_pt == False:
                lower_pt = True
                v_i = i
        # if irr_ends and lower_pt:
        #     if check_lower_pt(i):
        #         lower_pt = True
        #         v_i = i
        if irr_ends and lower_pt and i > v_i and check_break_point(i):
            # if any(x > 0 for x in b[i:i+5]):
            #     lower_pt = False
            #     continue
            
            ans.append(date_times[i])
            cal_indices.append(i+1)
            irr_ends = False
            lower_pt = False
            irr_detected = False

        if check_if_valley_point(i):
            ans.pop()
            cal_indices.pop()
            irr_ends = True
            lower_pt = True
            v_i = i
    return ans, cal_indices
import numpy as np
def get_calibration_value(values, indices):
    if len(indices) < 3:
        return None
    # min_ = np.percentile(values[-10*24:], 10)
    # max_ = np.percentile(values, 90)
    # values = np.array(values)
    # denominator = max_ - min_ or 1
    # values = (values - min_) / denominator
    selected = [values[i] for i in indices][-3:]

    # n = len(selected)
    # selected = sorted(selected)
    # i1 = int(0.1*n)
    # i2 = int(0.9*n)
    # if len(selected[i1:i2+1]) < 2:
    #     return None
    # std = stdev(selected[i1:i2+1])
    # avg = mean(selected[i1:i2+1])
    # return std*100/((avg-min_) or 1)
    return np.mean(selected)

import numpy as np
from scipy.stats import linregress, siegelslopes

def find_calibration_points3(values, times, window_size=5, threshold=2, threshold_2=2, ps=-0.001, cs=0):
    calibration_times = []
    slopes = []
    calibration_indices = []
    n = len(values)
    irr_detected = False
    irr_idx = None
    cal_set = False
    for i in range(0+window_size, n-window_size-1, 1):
        if values[i] - values[i-1] > 0.01:
            irr_detected = True
            irr_idx = i
            continue
        prev_window = values[i-window_size:i]
        curr_window = values[i:i+window_size]
        t = np.arange(window_size)
        prev_slope = linregress(t, prev_window).slope
        # prev_slope, intercept = siegelslopes(y=prev_window, x=t)
        curr_slope = linregress(t, curr_window).slope
        # curr_slope, intercept = siegelslopes(y=curr_window, x=t)
        
        if irr_detected and \
        prev_slope <= 0 and curr_slope <= cs and abs(prev_slope) > threshold * abs(curr_slope):
            calibration_times.append(times[i])
            calibration_indices.append(i+1)
            irr_detected = False
            cal_set = True

            continue

        if cal_set and (not irr_detected) and (calibration_times) and\
        prev_slope <= ps and curr_slope <= cs and abs(prev_slope) > threshold_2 * abs(curr_slope) and i <= irr_idx + 24:
            calibration_times.pop()
            calibration_indices.pop()

            calibration_times.append(times[i])
            calibration_indices.append(i+1)
            continue

        if irr_detected (not cal_set) and (i >= irr_idx + 18):
            calibration_times.append(times[i])
            calibration_indices.append(i+1)
            irr_detected = False
            cal_set = True
            continue

    return calibration_times, calibration_indices

from scipy.stats import linregress
def find_calibration_points(values, times, window_size=5, threshold=2, threshold_2=2, ps=-0.001, cs=0):
    calibration_times = []
    slopes = []
    calibration_indices = []
    n = len(values)
    irr_detected = False
    irr_idx = None
    cal_set = False
    for i in range(window_size, n-window_size-1, 1):
        if values[i] - values[i-1] > 0.01:
            irr_detected = True
            irr_idx = i
            continue
        prev_window = values[i-window_size:i+1]
        curr_window = values[i:i+window_size+1]
        t = np.arange(window_size+1)
        prev_slope = linregress(t, prev_window).slope
        ## prev_slope, intercept = siegelslopes(y=prev_window, x=t)
        curr_slope = linregress(t, curr_window).slope
        ## curr_slope, intercept = siegelslopes(y=curr_window, x=t)
        
        if irr_detected and \
        prev_slope < 0 and curr_slope <= cs and abs(prev_slope) > threshold * abs(curr_slope):
            calibration_times.append(times[i])
            calibration_indices.append(i+1)
            irr_detected = False
            cal_set = True

            continue

        if cal_set and (not irr_detected) and (calibration_times) and\
        prev_slope < ps and curr_slope < cs and abs(prev_slope) > threshold_2 * abs(curr_slope) and i < irr_idx + 18:
            calibration_times.pop()
            calibration_indices.pop()

            calibration_times.append(times[i])
            calibration_indices.append(i+1)
            
    return calibration_times, calibration_indices


def sma(values, window_size):
    return np.convolve(values, np.ones(window_size)/window_size, mode='valid')


def find_calibration_in_window(a, date_times):
    pass


from statistics import mean, stdev
def find_calibration_points2(a, date_times):
    prev_slope = None
    prev_stdev = None
    cal_points = []
    window = 10
    for i in range(len(a)-window-1, window, -1):
        curr_slope = mean(a[i-window//2:i+window//2+1])
        curr_stdev = stdev(a[i-window//2:i+window//2+1])
        if prev_slope is not None and prev_stdev is not None:
            if curr_slope < 1.4*prev_slope and curr_stdev > 1.8*prev_stdev \
            and prev_slope < 0 and curr_slope < 0:
                cal_points.append(date_times[i])
        prev_slope = curr_slope
        prev_stdev = curr_stdev
    return cal_points

def percentile_min_max_scale(arr, lower_pct=25, upper_pct=75):
    lo = np.percentile(arr, lower_pct)
    hi = np.percentile(arr, upper_pct)
    denominator = hi - lo or 1
    return (arr - lo) / denominator

def split_on_irrigation_and_min_max_scaling(arr):
    pass
    split_indices = []
    for i in range(1,n-2,1):
        if arr[i] - arr[i-1] > 0.01:
            split_indices.append(i)
    

import numpy as np
import pandas as pd

def time_to_90(df, col="moisture", threshold=0.9, irr_increase=0.01):
    values = df[col].values
    times = df.index.to_numpy()
    results = []
    n = len(values)
    
    for i in range(1, n):
        if values[i] - values[i-1] > irr_increase:
            peak_idx = i
            peak_val = values[peak_idx]
            cutoff = threshold * peak_val
            t90 = None
            censored = True
            for j in range(peak_idx+1, n):
                if values[j] - values[j-1] > irr_increase:
                    break
                if values[j] < cutoff:
                    dt = (times[j] - times[peak_idx]) / np.timedelta64(1, "h")
                    t90 = dt
                    censored = False
                    break
            results.append({
                "event_time": times[peak_idx],
                "peak_value": peak_val,
                "t90": t90,
                "censored": censored
            })
    return pd.DataFrame(results)

import numpy as np
import pandas as pd

def time_to_90(df, col="I1", threshold=0.9, irr_increase=0.01):
    values = df[col].values
    times = df.index.to_numpy()
    n = len(values)

    results = pd.DataFrame(columns=["event_time", "peak_value", "t90", "censored"])
    
    i = 1
    while i < n:
        if values[i] - values[i-1] > irr_increase:
            start_idx = i
            
            peak_idx = start_idx
            peak_val = values[peak_idx]
            
            while i < n and values[i] >= values[i-1]:
                if values[i] > peak_val:
                    peak_idx = i
                    peak_val = values[peak_idx]
                i += 1
            
            cutoff = threshold * peak_val
            t90 = None
            censored = True
            
            for j in range(peak_idx+1, n):
                if values[j] < cutoff:
                    dt = (times[j] - times[peak_idx]) / np.timedelta64(1, "h")
                    t90 = dt
                    censored = False
                    break
            
            results.loc[len(results)] = {
                "event_time": times[peak_idx],
                "peak_value": peak_val,
                "t90": t90,
                "censored": censored
            }
            i += 1
        else:
            i += 1
    
    return results

import plotly.express as px

def plot_t90_histogram(df):
    print(f"Original dataframe shape: {df.shape}")
    print(f"Original t90 column info: {df['t90'].describe()}")
    
    df = df.dropna(subset=["t90"])
    print(f"After dropping NaN: {df.shape}")
    print(f"t90 values: {df['t90'].tolist()}")
    
    if df.empty or df["t90"].isna().all():
        print("DataFrame is empty or all NaN - creating empty histogram")
        fig = px.histogram(
            df,
            x="t90",
            title="Time-to-90% Histogram (No Data)",
            labels={"t90": "Time to 90% (hours)"},
            opacity=0.75
        )
    else:
        max_t90 = df["t90"].max()
        min_t90 = df["t90"].min()
        print(f"t90 range: {min_t90} to {max_t90}")
        
        if pd.isna(max_t90):
            nbins = 24
            print("Using default 24 bins")
        else:
            nbins = max(1, int(max_t90) + 1)
            print(f"Using {nbins} bins")
        
        fig = px.histogram(
            df,
            x="t90",
            nbins=nbins,
            title=f"Time-to-90% Histogram ({len(df)} events)",
            labels={"t90": "Time to 90% (hours)"},
            opacity=0.75
        )
    
    fig.update_layout(bargap=0.1)
    return fig



if __name__ == "__main__":
    tests = [
        [0]*4 + [0, 0, 11.4, 0, 0, 0, 0, 0, 0, -0.0041,0,0,0,0,0, -11.4, 0, 0, 0, 0] + [0]*4
    ]

    ans = find_calibration_points(tests[0], list(range(len(tests[0]))))
    cal_index = ans[0][-1]
    cal_value = tests[0][cal_index-2:cal_index+2]
    print(cal_value)