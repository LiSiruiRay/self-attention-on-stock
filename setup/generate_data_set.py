# Author: ray
# Date: 2/10/24
# Description:
import copy
import csv
import json
import os.path
from datetime import datetime, timedelta

from tqdm import tqdm

from config import DATE_FORMAT
from datatype.news_stock_data import NSData
from datatype.training_dataset import SPDS
from util.common import text_to_md5_hash, get_proje_root_path
import pytz

ns = NSData()
ns.load_data()

ns.save_data()

md = SPDS()

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


def get_target_vector(time_str_index, sorted_stock_list, sorted_stock_keys_with_index,
                      detect_price: str = "4. close", ):
    time_span_list = [5, 10, 15, 30, 60, 180]
    vector = []
    for i in time_span_list:
        index_next = time_str_index + int(i / 5)
        if index_next > len(sorted_stock_keys_with_index) - 1:
            vector.append(None)
            continue
        stock_price_next = sorted_stock_list[index_next][1][detect_price]
        stock_price_curr = sorted_stock_list[time_str_index][1][detect_price]
        # print(f"result stock_price_next: {stock_price_next}, stock_price_curr: {stock_price_curr}")
        change_rate = float((float(stock_price_next) - float(stock_price_curr)) / float(stock_price_curr)) * 1000.0
        vector.append(change_rate)
    return vector


with open(os.path.join(porje_root, "data/stock_list.json"), 'r') as f:
    sorted_stock_list = json.load(f)

with open(os.path.join(porje_root, "data/sorted_keys_with_index.json"), 'r') as f:
    sorted_keys_with_index = json.load(f)

US_e_timezone = pytz.timezone('US/Eastern')
data_list = []

# calculate the amount, for process bar
n = 2882
a = 22760 - 2631

# 17322646

t = (n / 2) * (2 * a + (n - 1))
pbar = tqdm(total=17322646)

text_id_list = []
time_effect_list = []

time_info_list_1 = []
time_info_list_2 = []
time_info_list_3 = []

target_vector_list_0 = []
target_vector_list_1 = []
target_vector_list_2 = []
target_vector_list_3 = []
target_vector_list_4 = []
target_vector_list_5 = []


def calculate_time_since_premarket_start(date_et: datetime):
    # date = self.get_date_from_index(index=index)
    #
    # # Convert to Eastern Time (ET)
    et_timezone = pytz.timezone('US/Eastern')
    # date_et = date.astimezone(et_timezone)

    # Get the date part of the datetime
    given_date_only = date_et.date()

    # Determine the start of the pre-market session
    if date_et.hour < 4:
        # Before the pre-market, find the last market day
        last_market_day = SPDS.get_last_market_day(given_date_only - timedelta(days=1))
    else:
        # During or after the pre-market on the same day
        last_market_day = SPDS.get_last_market_day(given_date_only)

    # Set the pre-market start time
    pre_market_start_et = datetime(last_market_day.year, last_market_day.month,
                                   last_market_day.day, 4, 0, 0, 0, et_timezone)

    # Calculate the difference in milliseconds
    time_since_session_start_s = (date_et - pre_market_start_et).total_seconds()
    return time_since_session_start_s


def get_time_in_a_day(date: datetime) -> float:
    # date = self.get_date_from_index(index=index)
    midnight_datetime = datetime(date.year, date.month, date.day, 0, 0, 0,
                                 tzinfo=date.tzinfo)
    difference = date - midnight_datetime
    seconds_since_midnight = difference.total_seconds()
    return seconds_since_midnight


base_dt = datetime(2023, 5, 1, 0, 0, 0, 0).replace(tzinfo=pytz.utc)
date_list = []
for date, key in ns.news_data_map.items():
    if '.' in date.split('+')[0]:
        format_str = "%Y-%m-%dT%H:%M:%S.%f%z"
    else:
        format_str = "%Y-%m-%dT%H:%M:%S%z"
    news_time = datetime.strptime(date, format_str)
    # news_time = news_time.astimezone(US_e_timezone)
    # del key['stock_data']["sentiment"]
    # del key['stock_data']["ds_id"]
    # del key['stock_data']["model_name"]
    # del key['stock_data']["num_of_tokens"]
    # del key['stock_data']["sentiment_token"]
    # del key['stock_data']["data"]
    # del key["index"]

    vector_list = []
    curr_stock_data = key["stock_data"]
    curr_stock_time_str = curr_stock_data["stock_time"]
    curr_stock_index = sorted_keys_with_index[curr_stock_time_str]

    time_since_base = (news_time - base_dt).total_seconds()
    time_in_a_day = get_time_in_a_day(news_time)
    time_since_premarket_start = calculate_time_since_premarket_start(news_time.astimezone(pytz.timezone('US/Eastern')))

    time_vec = [time_since_base, time_in_a_day, time_since_premarket_start]

    # for i in range(curr_stock_index, curr_stock_index + 10):
    for i in range(curr_stock_index, len(sorted_keys_with_index)):
        vector = get_target_vector(i, sorted_stock_list=sorted_stock_list,
                                   sorted_stock_keys_with_index=sorted_keys_with_index)
        stock_time_dt = datetime.strptime(sorted_stock_list[i][0], DATE_FORMAT)
        stock_time_dt = US_e_timezone.localize(stock_time_dt)
        stock_time_dt = stock_time_dt.astimezone(pytz.utc)
        time_difference = stock_time_dt - news_time
        time_difference_sec = time_difference.total_seconds()
        # copy_curr = copy.deepcopy(key)
        # copy_curr["time_effect"] = time_difference_sec / 1e5
        # copy_curr["target_vec"] = vector
        # copy_curr["detected_stock_time"] = sorted_stock_list[i][0]

        date_list.append(date)

        pbar.update(1)

        # data_list.append(copy_curr)
        time_info_list_1.append(time_vec[0])
        time_info_list_2.append(time_vec[1])
        time_info_list_3.append(time_vec[2])

        text_id_list.append(key['stock_data']["text_id"])

        time_effect_list.append(time_difference_sec / 1e5)

        target_vector_list_0.append(vector[0])
        target_vector_list_1.append(vector[1])
        target_vector_list_2.append(vector[2])
        target_vector_list_3.append(vector[3])
        target_vector_list_4.append(vector[4])
        target_vector_list_5.append(vector[5])

with open(os.path.join(porje_root, "data/dataset.csv"), 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the header (optional)
    writer.writerow(["date", 'text_id', 'time_v0',
                     'time_v1', 'time_v2',
                     'effect_time', "tar_0",
                     "tar_1", "tar_2", "tar_3",
                     "tar_4", "tar_5"])
    for row in zip(date_list,
                   text_id_list,
                   time_info_list_1, time_info_list_2, time_info_list_3,
                   time_effect_list,
                   target_vector_list_0,
                   target_vector_list_1,
                   target_vector_list_2,
                   target_vector_list_3,
                   target_vector_list_4,
                   target_vector_list_5):
        if row[11] is None:
            continue
        writer.writerow(row)

# with open(os.path.join(porje_root, "data/full_data.json"), 'w') as f:
#     json.dump(data_list, f, indent=4)
