# Author: ray
# Date: 2/10/24
# Description:
from datatype.news_stock_data import NSData

ns = NSData()
ns.load_data()

ns.save_data()
counter = 0
for date, stock_data in ns.news_data_map.items():

    counter += 1