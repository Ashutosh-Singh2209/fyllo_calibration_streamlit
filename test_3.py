from test import get_field_data, pd
from test2 import find_calibration_points4

docs = get_field_data("plotno1335")

df = pd.DataFrame(docs)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp", ascending=True)

ts_list = df["timestamp"].iloc[:].to_list()

si = 30
ei = 49

vals = df['I1'].to_list()

print(find_calibration_points4(vals[si:ei+1], ts_list[si:ei+1]))