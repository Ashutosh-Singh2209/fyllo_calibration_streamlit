import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from get_data import get_plot_ids, get_field_data
from make_histogram import slope_histogram
from utils import find_calibration_points, get_calibration_value
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

st.set_page_config(page_title="Slope Histogram Clustering Analysis", layout="wide")

def diffs_from_sorted(values):
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return np.array([], dtype=float)
    diffs = np.diff(arr)
    return diffs

def get_histogram_frequencies(values):
    edges = np.arange(-0.02, 0.0000001, 0.0005)
    bins = np.concatenate(([-np.inf], edges, [np.inf]))
    data = values[np.array(values) < 0]
    
    if len(data) == 0:
        return None
    
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    if len(counts) == 0 or np.sum(counts) == 0:
        return None
    
    return counts, bin_centers

def get_histogram_features(values, remove_leftmost=True):
    edges = np.arange(-0.02, 0.0000001, 0.0005)
    bins = np.concatenate(([-np.inf], edges, [np.inf]))
    data = values[np.array(values) < 0]
    
    if len(data) == 0:
        return None, None
    
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    if remove_leftmost:
        counts = counts[1:]
        bin_centers = bin_centers[1:]
    
    if len(counts) == 0 or np.sum(counts) == 0:
        return None, None
    
    sorted_indices = np.argsort(counts)[::-1]
    largest_bin_idx = sorted_indices[0]
    second_largest_bin_idx = sorted_indices[1] if len(sorted_indices) > 1 else largest_bin_idx
    
    largest_bin_midpoint = bin_centers[largest_bin_idx]
    second_largest_bin_midpoint = bin_centers[second_largest_bin_idx]
    
    return largest_bin_midpoint, second_largest_bin_midpoint

def analyze_plot_data(plot_id):
    docs = get_field_data(plot_id)
    if not docs:
        return []
    
    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", ascending=True)
    
    plot_features = []
    
    for col in ["I1", "I2"]:
        if col in df.columns:
            df_col = df.dropna(subset=[col])
            if len(df_col) < 21:
                continue
                
            values = df_col[col].to_numpy()
            diffs = diffs_from_sorted(values)
            
            if len(diffs) > 20:
                histogram_data = get_histogram_frequencies(diffs)
                if histogram_data is not None:
                    counts, bin_centers = histogram_data
                    features = {
                        'plot_id': plot_id,
                        'sensor_type': col
                    }
                    for i, (count, center) in enumerate(zip(counts, bin_centers)):
                        features[f'bin_{i}_freq'] = count
                        features[f'bin_{i}_center'] = center
                    plot_features.append(features)
    
    return plot_features

st.title("Slope Histogram Clustering Analysis")
st.markdown("Analyzing histogram features from I1 and I2 slope data across multiple plots")


col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Load and Analyze Data"):
        with st.spinner("Loading plot data..."):
            plot_ids = get_plot_ids()
            st.info(f"Found {len(plot_ids)} plots to analyze")
            
            all_features = []
            plot_info = []
            
            progress_bar = st.progress(0)
            for i, plot_id in enumerate(plot_ids):
                plot_features = analyze_plot_data(plot_id)
                if plot_features:
                    all_features.extend(plot_features)
                    plot_info.extend([plot_id] * len(plot_features))
                
                progress_bar.progress((i + 1) / len(plot_ids))
            
            if all_features:
                df_features = pd.DataFrame(all_features)
                df_features.to_csv('histogram_frequencies.csv', index=False)
                st.success(f"Successfully analyzed {len(df_features)} plots with valid features and saved to CSV")
                
                csv_data = df_features.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name='histogram_frequencies.csv',
                    mime='text/csv'
                )

with col2:
    if st.button("Load from Local CSV"):
        try:
            df_features = pd.read_csv('histogram_frequencies.csv')
            st.success(f"Loaded {len(df_features)} features from local CSV")
        except FileNotFoundError:
            st.error("No local CSV file found. Please run 'Load and Analyze Data' first.")

with col3:
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None:
        try:
            df_features = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df_features)} features from uploaded CSV")
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")

if 'df_features' in locals():
    st.subheader("Feature Summary")
    st.dataframe(df_features.describe())
    
    freq_cols = [col for col in df_features.columns if col.startswith('bin_') and col.endswith('_freq')]
    
    if len(freq_cols) >= 2:
        st.subheader("2D Feature Visualization")
        st.markdown("Select histogram frequency bins to compare:")
        
        col_x, col_y = st.columns(2)
        with col_x:
            x_feature = st.selectbox("X-axis feature", freq_cols, index=0)
        with col_y:
            y_feature = st.selectbox("Y-axis feature", freq_cols, index=1)
        
        fig_scatter = px.scatter(
            df_features, 
            x=x_feature, 
            y=y_feature,
            hover_data=['plot_id'],
            title=f"{x_feature} vs {y_feature}",
            labels={x_feature: x_feature, y_feature: y_feature},
            width=600,
            height=800
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.subheader("Additional Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Show Top Frequency Bins"):
                freq_summary = df_features[freq_cols].sum().sort_values(ascending=False)
                st.bar_chart(freq_summary.head(10))
                st.caption("Bins with highest total frequencies across all plots")
        
        with col2:
            if st.button("Show Sensor Comparison"):
                sensor_summary = df_features.groupby('sensor_type')[freq_cols].mean()
                st.dataframe(sensor_summary.round(2))
                st.caption("Average frequency per bin by sensor type")
        
        st.subheader("Detailed Results")
        st.dataframe(df_features)
        
    else:
        st.warning("Need at least 2 features for 2D visualization")

st.markdown("---")
st.markdown("### How it works:")
st.markdown("""
1. **Data Loading**: Loads plot data from MongoDB for NERO_INFINITY_UNIT devices
2. **Slope Calculation**: Computes consecutive differences (slopes) for I1 and I2 values
3. **Histogram Analysis**: Creates histograms of negative slopes with fixed bins from -0.02 to 0.0
4. **Frequency Extraction**: 
   - Saves frequency counts for all remaining bins (including leftmost)
   - Each plot/sensor combination becomes a row with all bin frequencies
5. **2D Visualization**: Compare any two histogram frequency bins to identify patterns
6. **Data Export/Import**: Download CSV with all frequencies or upload existing data
""")
