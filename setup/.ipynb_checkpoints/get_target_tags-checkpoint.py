# Author: ray
# Date: 1/22/24
# Description:
import json
from collections import OrderedDict

with open('../data/news_to_stock_valid_time.json', 'r') as file:
    news_to_stock_data = json.load(file)

with open('../data/raw_stock_data.json', 'r') as file:
    stock_data = json.load(file)

stock_data = stock_data["Time Series (5min)"]

sorted_keys = sorted(stock_data, reverse=False)  # Use reverse=True if you want latest times first

# Step 3: Create a new ordered dictionary
sorted_data = OrderedDict()

for i, k in enumerate(sorted_keys):
    sorted_data[k] = {"stock_data": stock_data[k],
                      "index": i}


# If you need to use the sorted data, you can iterate over sorted_data
# for datetime_str, values in sorted_data.items():
#     print(datetime_str, values)

def get_target_vector(time_str: str, detect_price: str = "4. close"):
    curr_time_str_index = sorted_data[time_str]["index"]
    time_span_list = [5, 10, 15, 30, 60, 180]
    vector = []
    for i in time_span_list:
        index_next = curr_time_str_index + int(i / 5)
        next_time_str = sorted_keys[index_next]
        stock_price_next = stock_data[next_time_str][detect_price]
        stock_price_curr = stock_data[time_str][detect_price]
        print(f"time str: {time_str}")
        print(f"result stock_price_next: {stock_price_next}, stock_price_curr: {stock_price_curr}")
        change_rate = float((float(stock_price_next) - float(stock_price_curr)) / float(stock_price_curr)) * 100.0
        vector.append(change_rate)

    return vector
    # index_5min = curr_time_str_index + int(5 / 5)
    # index_10min = curr_time_str_index + int(10 / 5)
    # index_15min = curr_time_str_index + int(15 / 5)
    # index_30min = curr_time_str_index + int(30 / 5)
    # index_60min = curr_time_str_index + int(60 / 5)
    # index_180min = curr_time_str_index + int(180 / 5)


id_to_vec = dict()
for i, (k, v) in enumerate(news_to_stock_data.items()):
    print(f"-----------------check key: {k}-------------")
    time_stamp = v
    id_to_vec[k] = get_target_vector(time_str=time_stamp)

with open('../data/news_to_target_tag.json', 'w') as file:
    json.dump(id_to_vec, file, indent=4)
