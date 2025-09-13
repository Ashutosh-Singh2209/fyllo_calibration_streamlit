import os
from datetime import datetime, timedelta
from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

load_dotenv(find_dotenv(), override=True)

def get_mongo_collections():
    client = MongoClient(os.getenv('FYLLO_MONGO_URI'))
    db = client['database']
    return db['device'], db['FinalFieldData']

def fetch_plot_ids(device_collection):
    filter_query = {
        'deviceType': 'NERO_INFINITY_UNIT',
        'installationDate': {'$gte': datetime(2025, 7, 25)},
        'isAssigned': True
    }
    return [d['plotId'] for d in device_collection.find(filter_query, {'plotId': 1, '_id': 0})]

def fetch_live_data(field_data_collection, plot_id):
    filter_query = {
        'plotId': plot_id,
        'timestamp': {'$gte': datetime(2025, 7, 25)}
    }
    projection = {
        '_id': 1,
        'deviceId': 1,
        'plotId': 1,
        'farmUserId': 1,
        'timestamp': 1,
        'I1': 1,
        'I2': 1
    }
    return list(field_data_collection.find(filter_query, projection))

from make_histogram import slope_bin_count as compute_slope_hist_frequencies
from make_histogram import bin_counts as compute_hist_frequencies

def iter_10d_windows(start_ts, end_ts):
    current = start_ts
    while current < end_ts:
        next_ts = current + timedelta(days=10)
        yield current, min(next_ts, end_ts)
        current = next_ts

def process_plot(plot_id, docs):
    if not docs:
        return []
    df = pd.DataFrame(docs)
    if 'timestamp' not in df.columns:
        return []
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    if df.empty:
        return []
    start_ts = df['timestamp'].min().floor('D')
    end_ts = (df['timestamp'].max() + pd.Timedelta(seconds=1)).ceil('D')
    results = []
    for win_start, win_end in iter_10d_windows(start_ts, end_ts):
        mask = (df['timestamp'] >= win_start) & (df['timestamp'] < win_end)
        win_df = df.loc[mask]
        if win_df.empty:
            continue
        for metric in ['I1', 'I2']:
            if metric not in win_df.columns:
                continue
            freqs = compute_hist_frequencies(win_df, metric)
            slope_freqs = compute_slope_hist_frequencies(win_df, metric)
            if freqs is None:
                continue
            row = {
                'plot_id': plot_id,
                'date_start': win_start,
                'date_end': win_end,
                'metric': metric
            }
            for i, f in enumerate(freqs):
                row[f'bin_{i}'] = f
            for i, f in enumerate(slope_freqs):
                row[f'slope_bin_{i}'] = f
            results.append(row)
    return results

def main():
    device_collection, field_data_collection = get_mongo_collections()
    plot_ids = sorted(fetch_plot_ids(device_collection))
    random.shuffle(plot_ids)
    plot_ids = plot_ids[:600]
    all_rows = []
    for plot_id in tqdm(plot_ids, desc="Processing plots"):
        docs = fetch_live_data(field_data_collection, plot_id)
        if len(docs) < 24*9:
            continue
        rows = process_plot(plot_id, docs)
        all_rows.extend(rows)
    if not all_rows:
        df_out = pd.DataFrame(columns=['plot_id', 'date_start', 'date_end', 'metric'])
    else:
        df_out = pd.DataFrame(all_rows)
    out_path = os.path.join(os.getcwd(), 'histogram_features.csv')
    df_out.to_csv(out_path, index=False)

if __name__ == '__main__':
    main()


