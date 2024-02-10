# Author: ray
# Date: 2/10/24
# Description:
import json
import os.path

from util.common import get_proje_root_path

proje_root = get_proje_root_path()
with open(os.path.join(proje_root, f"data/raw_stock_data.json"), 'r') as f:
    stock_data = json.load(f)

stock_data = stock_data["Time Series (5min)"]
stock_keys = list(stock_data.keys())
sorted_keys = sorted(stock_keys, reverse=False)
sorted_keys_with_index = {}
for i, j in enumerate(sorted_keys):
    sorted_keys_with_index[j] = i

with open(os.path.join(proje_root, f"data/sorted_stock_keys.json"), 'w') as f:
    json.dump(sorted_keys, f, indent=4)

with open(os.path.join(proje_root, f"data/sorted_keys_with_index.json"), 'w') as f:
    json.dump(sorted_keys_with_index, f, indent=4)

reversed_stock_data = []

for i, (k, v) in enumerate(stock_data.items()):
    curr_time = sorted_keys[i]
    data_list = [k, v, {"index": i}]
    reversed_stock_data.append(data_list)

with open(os.path.join(proje_root, f"data/stock_list.json"), 'w') as f:
    json.dump(reversed_stock_data, f, indent=4)