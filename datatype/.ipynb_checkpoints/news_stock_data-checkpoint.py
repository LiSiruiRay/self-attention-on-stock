# Author: ray
# Date: 1/22/24
# Description:
import json
import os
from collections import OrderedDict
from datetime import datetime

from util.common import get_proje_root_path


class NSData:
    news_list: list  # to access from index, list of time stamp
    news_data_map: OrderedDict  # to maintain the order
    news_data_path: str
    news_to_timedata: dict

    def __init__(self):
        project_path = get_proje_root_path()
        self.news_data_path = os.path.join(project_path, "data/news_data_valid_time_valid_range.json")

    def load_data(self):
        news_data = {}
        time_tamp_to_news_data = {}
        news_to_timedata = {}

        with open(self.news_data_path, 'r') as file:
            news_data = json.load(file)
        for index, (i, each) in enumerate(news_data.items()):
            each_date_str = each["timestamp"]
            news_to_timedata[each["data"]] = each_date_str

            time_tamp_to_news_data[each_date_str] = each

        sorted_keys = sorted(time_tamp_to_news_data, reverse=False)
        self.news_list = sorted_keys # time
        sorted_news_data = OrderedDict()
        for i, k in enumerate(sorted_keys):
            sorted_news_data[k] = {"stock_data": time_tamp_to_news_data[k],
                                   "index": i}
        self.news_data_map = sorted_news_data
        self.news_to_timedata = news_to_timedata

    def get_timestr_from_text_data(self, text_data: str):
        time_str = self.news_to_timedata[text_data]
        return time_str

    def get_index_from_text_data(self, text_data: str):
        time_str = self.news_to_timedata[text_data]
        return self.news_data_map[time_str]["index"]

    def __len__(self):
        return len(self.news_list)

    def __getitem__(self, index):
        time_key = self.news_list[index]
        to_return = self.news_data_map[time_key]["stock_data"]
        return to_return
