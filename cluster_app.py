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
                largest_mid, second_mid = get_histogram_features(diffs)
                if largest_mid is not None:
                    features = {
                        'plot_id': plot_id,
                        'sensor_type': col,
                        'largest_bin': largest_mid,
                        'second_bin': second_mid
                    }
                    plot_features.append(features)
    
    return plot_features

st.title("Slope Histogram Clustering Analysis")
st.markdown("Analyzing histogram features from I1 and I2 slope data across multiple plots")


col1, col2 = st.columns(2)
with col1:
    if st.button("Load and Analyze Data"):
        with st.spinner("Loading plot data..."):
            plot_ids = get_plot_ids()
            st.info(f"Found {len(plot_ids)} plots to analyze")
            
            all_features = []
            plot_info = []
            
            progress_bar = st.progress(0)
            for i, plot_id in enumerate(plot_ids[:50]):
                plot_features = analyze_plot_data(plot_id)
                if plot_features:
                    all_features.extend(plot_features)
                    plot_info.extend([plot_id] * len(plot_features))
                
                progress_bar.progress((i + 1) / 50)
            
            if all_features:
                df_features = pd.DataFrame(all_features)
                df_features.to_csv('histogram_features.csv', index=False)
                st.success(f"Successfully analyzed {len(df_features)} plots with valid features and saved to CSV")
                
                csv_data = df_features.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name='histogram_features.csv',
                    mime='text/csv'
                )

with col2:
    if st.button("Load from CSV"):
        try:
            df_features = pd.read_csv('histogram_features.csv')
            st.success(f"Loaded {len(df_features)} features from CSV")
        except FileNotFoundError:
            st.error("No CSV file found. Please run 'Load and Analyze Data' first.")

if 'df_features' in locals():
    st.subheader("Feature Summary")
    st.dataframe(df_features.describe())
    
    feature_cols = [col for col in df_features.columns if col not in ['plot_id', 'sensor_type']]
    
    if len(feature_cols) >= 2:
        st.subheader("2D Feature Visualization")
        
        col_x, col_y = st.columns(2)
        with col_x:
            x_feature = st.selectbox("X-axis feature", feature_cols, index=0)
        with col_y:
            y_feature = st.selectbox("Y-axis feature", feature_cols, index=1)
        
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
        
        st.subheader("Detailed Results")
        st.dataframe(df_features)
        
    else:
        st.warning("Need at least 2 features for 2D visualization")

st.markdown("---")
st.markdown("### How it works:")
st.markdown("""
1. **Data Loading**: Loads plot data from MongoDB for NERO_INFINITY_UNIT devices
2. **Slope Calculation**: Computes consecutive differences (slopes) for I1 and I2 values
3. **Histogram Analysis**: Creates histograms of negative slopes with bins from -0.02 to 0.0
4. **Feature Extraction**: 
   - Removes the leftmost bin (most negative values)
   - Finds the largest and second largest bins
   - Extracts their midpoints as features
5. **2D Visualization**: Plots features in 2D space for pattern recognition
6. **Clustering**: Applies K-Means or DBSCAN clustering to identify groups
""")
