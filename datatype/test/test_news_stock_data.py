# Author: ray
# Date: 1/22/24

import unittest

from datatype.news_stock_data import NSData


class MyTestCaseDataLoaderNewsStockData(unittest.TestCase):
    def test_reading(self):
        nsd = NSData()
        nsd.load_data()
        print(f"check first: {nsd[0]}")


if __name__ == '__main__':
    unittest.main()
