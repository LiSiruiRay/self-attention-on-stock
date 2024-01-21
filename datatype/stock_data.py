# Author: ray
# Date: 1/22/24
# Description:
from dataclasses import dataclass
from typing import Any, Set
from datetime import datetime, timedelta


@dataclass
class DataPoint:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class StockData:
    meta: Any
    time_series_data: Set[DataPoint]


class BlurDatetime(datetime):
    def __eq__(self, other):
        if isinstance(other, datetime):
            difference = abs(self - other)
            return difference <= timedelta(minutes=5)
