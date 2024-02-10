
# Input data
1. feature of passage: passage vector
2. feature of time in relate to news: 
   1. absolute time (in relative to 2023 July 1st)
   2. time in a day
   3. time since market started
3. How is the stock data in relate to the news: how long after the news happened are we detecting
   1. For example: a news happened at 11:59pm, yet stock start at 9 (next day), that is 9 hours and 1 min after news happened

# Target Tag
A vector of fluctuation (percentage) of 

[
5min,
10min,
15min,
30min,
1h,
3h
]

if there is data dis continuity, I will just use the consecutive next ones.
For example, I have data of 1th 19:55, the 5min will just be 2th first data.

Design doc can be found under DesignDoc.md 