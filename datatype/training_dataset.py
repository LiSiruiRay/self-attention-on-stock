# Author: ray
# Date: 1/22/24
# Description:
import json
import os.path
from datetime import datetime, timedelta

import torch
from torch.utils.data import Dataset

from config import DATE_FORMAT
from datatype.news_stock_data import NSData
import pytz

from util.common import text_to_md5_hash, get_proje_root_path


class Mydataset(Dataset):
    sorted_news_data: NSData
    zero_time_stamp: datetime  # when is the base line
    news_to_stock: dict
    news_to_target_tag_data: dict
    use_reduced_passage_vec: bool

    @staticmethod
    def get_news_to_stock(file_name: str = "news_to_stock_valid_time.json"):
        project_root = get_proje_root_path()
        data_path = os.path.join(project_root, f"data/{file_name}")
        with open(data_path, 'r') as file:
            stock_data = json.load(file)
        return stock_data

    def load_news_to_target_tag_data(self):
        project_root = get_proje_root_path()
        file_name = "news_to_target_tag.json"
        data_path = os.path.join(project_root, f"data/{file_name}")
        with open(data_path, 'r') as file:
            news_to_target_tag_data = json.load(file)
        return news_to_target_tag_data

    def __len__(self):
        return len(self.sorted_news_data)

    def __init__(self,
                 zero_time_stamp: datetime =
                 datetime(2023, 5, 1, 0, 0, 0, 0)
                 .replace(tzinfo=pytz.utc),
                 use_reduced_passage_vec = True):
        """
        Only initialization needed
        :param zero_time_stamp:
        """
        self.sorted_news_data = NSData()
        self.sorted_news_data.load_data()
        self.zero_time_stamp = zero_time_stamp
        self.news_to_stock = Mydataset.get_news_to_stock()
        self.news_to_target_tag_data = self.load_news_to_target_tag_data()
        self.use_reduced_passage_vec = use_reduced_passage_vec

    def __getitem__(self, index: int):
        news_data = self.sorted_news_data[index]
        text_data = news_data['data']
        text_id = text_to_md5_hash(text_data)
        text_id = f"{text_id}_reduced.pt" if self.use_reduced_passage_vec else f"{text_id}.pt"
        project_root = get_proje_root_path()
        text_id = os.path.join(project_root, f"data/text_vector/{text_id}")
        passage_vec = torch.load(text_id).float()
        time_since_base = self.get_time_since_base(index=index)
        time_in_a_day = self.get_time_in_a_day(index=index)
        time_since_pre_market_start = self.calculate_time_since_premarket_start(index=index)
        time_of_effect = self.get_time_for_effect_on_stock(index=index)

        normalized_data = Mydataset.normalize_time_data(time_since_base=time_since_base,
                                                        time_in_a_day=time_in_a_day,
                                                        time_since_pre_market_start=time_since_pre_market_start)
        time_vec = torch.tensor(normalized_data)
        time_of_effect_normalized = time_of_effect / (48*60*60*1000)

        target_vec = self.get_target_vec(index=index)

        passage_vec = passage_vec.squeeze(0)

        return (passage_vec.float(), time_vec, torch.tensor([time_of_effect_normalized])), target_vec.float()

    def get_target_vec(self, index: int) -> torch.tensor:
        news_data = self.sorted_news_data[index]
        text_data = news_data['data']
        text_id = text_to_md5_hash(text_data)
        target_vec = self.news_to_target_tag_data[text_id]
        return torch.tensor(target_vec)

    @staticmethod
    def normalize_time_data(time_since_base: float,
                            time_in_a_day: float,
                            time_since_pre_market_start: float):
        data_list = [time_since_base, time_in_a_day, time_since_pre_market_start]
        min_data = min(data_list)
        max_data = max(data_list)
        data_list = [(i - min_data) / (max_data - min_data) for i in data_list]
        return data_list

    def get_date_from_index(self, index: int) -> datetime:
        news_data = self.sorted_news_data[index]
        date_str = news_data['timestamp']
        if '.' in date_str.split('+')[0]:
            format_str = "%Y-%m-%dT%H:%M:%S.%f%z"
        else:
            format_str = "%Y-%m-%dT%H:%M:%S%z"
        date = datetime.strptime(date_str, format_str)
        return date

    def get_time_for_effect_on_stock(self, index: int) -> int:
        date_news_dt = self.get_date_from_index(index=index)
        news_data = self.sorted_news_data[index]
        text_data = news_data['data']
        text_id = text_to_md5_hash(text_data)
        date_stock = self.news_to_stock[text_id]
        date_stock_dt = datetime.strptime(date_stock, DATE_FORMAT)
        et_timezone = pytz.timezone('US/Eastern')
        date_stock_dt = et_timezone.localize(date_stock_dt)

        date_stock_dt_utc = date_stock_dt.astimezone(pytz.utc)

        difference = date_stock_dt_utc - date_news_dt

        milliseconds_since_midnight = int(difference.total_seconds() * 1000 + difference.microseconds / 1000)
        return milliseconds_since_midnight

    def get_time_since_base(self, index: int) -> int:
        date = self.get_date_from_index(index=index)
        difference = date - self.zero_time_stamp
        difference_in_milliseconds = int(difference.total_seconds() * 1000 + difference.microseconds / 1000)
        return difference_in_milliseconds

    def get_time_in_a_day(self, index: int) -> int:
        date = self.get_date_from_index(index=index)
        midnight_datetime = datetime(date.year, date.month, date.day, 0, 0, 0,
                                     tzinfo=date.tzinfo)
        difference = date - midnight_datetime
        milliseconds_since_midnight = int(difference.total_seconds() * 1000 + difference.microseconds / 1000)
        return milliseconds_since_midnight

    @staticmethod
    def is_weekday(d):
        """ Check if the given date is a weekday """
        return d.weekday() < 5  # Monday is 0, Sunday is 6

    @staticmethod
    def get_last_market_day(d):
        """ Get the last market day. If the given day is a market day, return it.
            Otherwise, return the last weekday before it. """
        while not Mydataset.is_weekday(d):
            d -= timedelta(days=1)
        return d

    def calculate_time_since_premarket_start(self, index: int):
        date = self.get_date_from_index(index=index)

        # Convert to Eastern Time (ET)
        et_timezone = pytz.timezone('US/Eastern')
        date_et = date.astimezone(et_timezone)

        # Get the date part of the datetime
        given_date_only = date_et.date()

        # Determine the start of the pre-market session
        if date_et.hour < 4:
            # Before the pre-market, find the last market day
            last_market_day = Mydataset.get_last_market_day(given_date_only - timedelta(days=1))
        else:
            # During or after the pre-market on the same day
            last_market_day = Mydataset.get_last_market_day(given_date_only)

        # Set the pre-market start time
        pre_market_start_et = datetime(last_market_day.year, last_market_day.month,
                                       last_market_day.day, 4, 0, 0, 0, et_timezone)

        # Calculate the difference in milliseconds
        time_since_session_start_ms = (date_et - pre_market_start_et).total_seconds() * 1000
        return time_since_session_start_ms
