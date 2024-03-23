import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
import time


client = Client(n_workers=4) #change number of workers based on test
client #ptints the specification of client

#computation
start_time = time.time()
x = da.random.random((10000, 10000, 10), chunks=(1000, 1000, 5)) #generate two array with 1000 by 1000 by 10 randomly
y = da.random.random((10000, 10000, 10), chunks=(1000, 1000, 5))
z = (da.arcsin(x) + da.arccos(y)).sum(axis=(1, 2)) #computaation
z.persist()#store it in memory so that it is readily available
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

















#client.close()