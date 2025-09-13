import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def histogram(df: pd.DataFrame, col: str):
    edges = np.arange(0, 1.5000001, 0.005)
    bins = np.concatenate(([-np.inf], edges, [np.inf]))
    data = df[col].dropna().values

    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    labels = []
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        if np.isneginf(left):
            labels.append(f'< {right:.3f}')
        elif np.isposinf(right):
            labels.append(f'>= {left:.3f}')
        else:
            labels.append(f'{left:.3f}–{right:.3f}')

    df_hist = pd.DataFrame({'bin': labels, 'count': counts})

    fig = px.bar(df_hist, x='bin', y='count')
    fig.update_layout(xaxis_tickangle=45, xaxis_title=col, yaxis_title='Count', title=f'Histogram of {col} with −∞ and +∞ bins')
    return fig

def slope_histogram(values):
    edges = np.arange(-0.02, 0.0000001, 0.0005)
    bins = np.concatenate(([-np.inf], edges, [np.inf]))
    data = values[np.array(values) < 0]

    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    labels = []
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        if np.isneginf(left):
            labels.append(f'< {right:.5f}')
        elif np.isposinf(right):
            labels.append(f'>= {left:.5f}')
        else:
            labels.append(f'{left:.5f}–{right:.5f}')

    df_hist = pd.DataFrame({'bin': labels, 'count': counts})

    fig = px.bar(df_hist, x='bin', y='count')
    fig.update_layout(xaxis_tickangle=45, xaxis_title="slope value range", yaxis_title='Count', title=f'Histogram of slopes')
    return fig

def bin_counts(df: pd.DataFrame, col: str):
    edges = np.arange(0, 1.5000001, 0.005)
    bins = np.concatenate(([-np.inf], edges, [np.inf]))
    data = df[col].dropna().values
    counts, _ = np.histogram(data, bins=bins)
    return counts.astype(float).tolist()

def slope_bin_count(df: pd.DataFrame, col: str):
    edges = np.arange(-0.02, 0.07000001, 0.0005)
    bins = np.concatenate(([-np.inf], edges, [np.inf]))
    differences = df[col].astype(float).diff().dropna().values
    counts, _ = np.histogram(differences, bins=bins)
    return counts.astype(float).tolist()

if '__main__' == __name__:
    import pandas as pd
    import numpy as np

    array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
    df = pd.DataFrame(array, columns=["A"])
    print(bin_counts(df, "A")+[None, None])