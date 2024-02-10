# Author: ray
# Date: 2/10/24
# Description:
import json
import os.path

from datatype.news_stock_data import NSData
from datatype.training_dataset import Mydataset
from util.common import text_to_md5_hash, get_proje_root_path

ns = NSData()
ns.load_data()

ns.save_data()

md = Mydataset()

for date, key in ns.news_data_map.items():
    stock_data = key["stock_data"]
    text_data = stock_data["data"]
    text_id = text_to_md5_hash(text_data)

    stock_date_str = md.news_to_stock[text_id]

    stock_data["text_id"] = text_id
    stock_data["stock_time"] = stock_date_str

# TODO: should not use dictionary since there might be two news happening at the same time
file_name = "extended_sorted_news_data_map.json"

porje_root = get_proje_root_path()
with open(os.path.join(porje_root, f"data/{file_name}"), 'w') as f:
    json.dump(ns.news_data_map, f, indent=4)


def get_target_vector(sorted_keys, sorted_data, time_str: str, detect_price: str = "4. close", ):
    curr_time_str_index = sorted_data[time_str]["stock_time"]
    time_span_list = [5, 10, 15, 30, 60, 180]
    vector = []
    for i in time_span_list:
        index_next = curr_time_str_index + int(i / 5)
        next_time_str = sorted_keys[index_next]
        stock_price_next = stock_data[next_time_str][detect_price]
        stock_price_curr = stock_data[time_str][detect_price]
        print(f"time str: {time_str}")
        print(f"result stock_price_next: {stock_price_next}, stock_price_curr: {stock_price_curr}")
        change_rate = float((float(stock_price_next) - float(stock_price_curr)) / float(stock_price_curr)) * 1000.0
        vector.append(change_rate)

    return vector