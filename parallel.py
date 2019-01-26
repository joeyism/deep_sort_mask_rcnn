from tqdm import tqdm
import multiprocessing as mp
import time
import numpy as np

cores = mp.cpu_count()

def parallelize(data, func):
    with mp.Pool(processes=cores) as p:
        result = list(p.imap(func, data))
    return result

def tqdm_parallelize(data, func, desc="Running in parallel", unit="it"):
    with mp.Pool(processes=cores) as p:
        result = list(tqdm(p.imap(func, data), desc=desc, total=len(data), unit=unit))

    return result

def parallelize_cleaned(data, func):
    result = parallelize(data, func)
    return [ val for val in result if val is not None]


