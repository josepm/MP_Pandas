import pandas as pd
import numpy as np
import sys
import os
home = os.path.expanduser("~")
sys.path.append(home + '/code')
import pandas_mp.mp_generic as mp
from datetime import datetime
import pytz
import time

EPOCH = datetime(1970, 1, 1, 0, 0, 0, tzinfo=pytz.utc)


def time_now(epoch=EPOCH):
    tt = datetime.utcnow().replace(tzinfo=pytz.utc)
    return (tt - epoch).total_seconds()


def timing(t0):
    t1 = time_now()
    return t1, t1 - t0


def apply_func(df, wait, work):
    time.sleep(wait)
    time.sleep(work * len(df))
    g = pd.DataFrame({'len': [len(df)]})
    return g


df_len = 100
high_1 = 2
high_2 = 3
df = pd.DataFrame({'gb1': np.random.randint(low=0, high=high_1, size=df_len),
                   'gb2': np.random.randint(low=0, high=high_2, size=df_len),
                   'a': np.random.normal(size=df_len)})

gb_cols = ['gb1', 'gb2']
args = [0.01, 0.001]  # wait, work

t0 = time_now()
z = df.groupby(gb_cols).apply(apply_func, *args)
timing(t0)


mp_args = {'n_cpus': 6, 'queue': True, 'n_queues': None}

t0 = time_now()
z = mp.mp_groupby(df, gb_cols, apply_func, *args, **mp_args)
timing(t0)
