import dask.dataframe as dd
%%time
df = dd.read_csv('train.csv') #900 MB with close to 950,000 observations and 119 features
df.head(3)