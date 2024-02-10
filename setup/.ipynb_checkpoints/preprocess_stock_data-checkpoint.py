# Author: ray
# Date: 1/22/24
# Description:

import json
from datetime import datetime, timedelta

from config import DATE_FORMAT
from util.common import text_to_md5_hash
import pytz

est_timezone = pytz.timezone('US/Eastern')


def round_up_to_next_five_min(dt: datetime):
    # If the time is already at a 5-minute interval, return it as is
    if dt.minute % 5 == 0 and dt.second == 0:
        return dt

    # Otherwise, round up to the next 5-minute interval
    rounded_dt = dt + timedelta(minutes=5)
    return rounded_dt.replace(second=0, microsecond=0) - timedelta(minutes=rounded_dt.minute % 5)


def get_key_in_stock_data(dt: datetime) -> str:
    round_up_dt = round_up_to_next_five_min(dt)
    round_up_dt = round_up_dt.astimezone(est_timezone)
    round_up_dt_str = round_up_dt.strftime(DATE_FORMAT)
    non_market = False  # some news happened not during market time
    max_interval = 30 * (60/5) * 3 # when time is too new and no new stock data yet
    counter = 0
    while round_up_dt_str not in stock_data["Time Series (5min)"]:
        non_market = True
        round_up_dt += timedelta(minutes=5)
        round_up_dt_str = round_up_dt.strftime(DATE_FORMAT)
        counter += 1
        if counter > max_interval:
            return "no stock data"
    if non_market:
        return f"{round_up_dt.strftime(DATE_FORMAT)}_nonmarket"
    return round_up_dt.strftime(DATE_FORMAT)


with open('../data/raw_stock_data.json', 'r') as file:
    stock_data = json.load(file)

with open('../data/news_data_valid_time.json', 'r') as file:
    news_data = json.load(file)

text_id_to_stock_data_key = dict()

for index, (i, each) in enumerate(news_data.items()):
    each_text = each['data']
    each_id = text_to_md5_hash(each_text)
    each_date_str = each["timestamp"]
    if '.' in each_date_str.split('+')[0]:
        format_str = "%Y-%m-%dT%H:%M:%S.%f%z"
    else:
        format_str = "%Y-%m-%dT%H:%M:%S%z"
    each_date = datetime.strptime(each_date_str, format_str)
    stock_data_key = get_key_in_stock_data(dt=each_date)
    split_key = stock_data_key.split('_')
    # if "nonmarket" in split_key:
    #     each_id = f"{each_id}_nonmarket"

    if "no stock data" == split_key[0]:
        each_id = f"{each_id}_nonstockdata"
        # print(f"each_id: {each_text}----\n\n {each}")
        continue
    if "2024-01-19 18" in split_key[0] or "2024-01-19 19" in split_key[0]:
        print(f"too late ones: {each}")
        continue
    text_id_to_stock_data_key[each_id] = split_key[0]
    print(f"finished filtering : {index}/{len(news_data)}")

with open('../data/news_to_stock_valid_time.json', 'w') as file:
    json.dump(text_id_to_stock_data_key, file, indent=4)


def get_middle_time(d_1: datetime, d_2: datetime) -> datetime:
    difference = d_1 - d_2
    middle_datetime = d_2 + (difference / 2)
    return middle_datetime
