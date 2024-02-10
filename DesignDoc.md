# AI on stock design doc

# Pipeline

## Set Up

1. get news
2. get stock data
3. preprocess both
    1. get valid time range
    2. prepare for the id, vectorization, metadata, etc
4. get correlation
5. training process start

# Project structure

- Name Conventions
    
    Always use plaril
    
    Always n. + v. (ing)
    
    e.g. passagevectorizing (not vectorize passage, not passage vectorization)
    
- Interrfaces/
    - SetUpInterfaces/
        - set_up_strategy.py
        - GetNews/
            - (not now)
        - GetStockData/
            - (not now)
        - PreProcessSN/
            - 
    - CorrelationInterfaces
    - TrainingInterfaces