# Author: ray
# Date: 1/22/24
# Description:

import json
from datetime import datetime, timedelta

with open('../data/raw_stock_data.json', 'r') as file:
    stock_data = json.load(file)


def get_middle_time(d_1: datetime, d_2: datetime) -> datetime:
    difference = d_1 - d_2
    middle_datetime = d_2 + (difference / 2)
    return middle_datetime
