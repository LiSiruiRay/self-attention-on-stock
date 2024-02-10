# Author: ray
# Date: 2/10/24
# Description:
import json
import os.path
from collections import defaultdict
from datetime import datetime

import pytz

from config import DATE_FORMAT
from util.common import get_proje_root_path


def load_news_to_stock_valid_time():
    proje_root_path = get_proje_root_path()
    file_path = os.path.join(proje_root_path, "data/news_to_stock_valid_time.json")
    with open(file_path, 'r') as file:
        news_to_stock_valid_time = json.load(file)
        return news_to_stock_valid_time


def stored_sorted_keys(news_to_stock_valid_time):
    date_list = list(news_to_stock_valid_time.values())
    date_list.sort()
    return date_list


news_to_stock_valid_time = load_news_to_stock_valid_time()
date_list = stored_sorted_keys(news_to_stock_valid_time)
print(f"result: {date_list}")

time_2_dict = defaultdict(list)
for k, v in news_to_stock_valid_time.items():
    time_2_dict[v].append(k)

# print(time_2_dict)
l = 0
for k, v in time_2_dict.items():
    l = l + len(time_2_dict[k])

# print(l == len(news_to_stock_valid_time))

sorted_time_to_passage = []
for k, v in time_2_dict.items():
    for i in v:
        sorted_time_to_passage.append([k, i])

print(len(sorted_time_to_passage) == len(news_to_stock_valid_time))
sorted_passage_2_time = [[v, k] for [k, v] in sorted_time_to_passage]
print(sorted_passage_2_time)

proje_root_path = get_proje_root_path()
data_folder = os.path.join(proje_root_path, "data/")
with open(os.path.join(data_folder, "date_list.json", 'w')) as f:
    json.dump(date_list, f)

with open(os.path.join(data_folder, "date_list.json", 'w')) as f:
    json.dump(date_list, f)

with open(os.path.join(data_folder, "date_list.json", 'w')) as f:
    json.dump(date_list, f)

with open(os.path.join(data_folder, "date_list.json", 'w')) as f:
    json.dump(date_list, f)