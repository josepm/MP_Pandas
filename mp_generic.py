"""
Generic multiprocessing support for Pandas calls of the form:
            df_out = df_in.groupby(gb_cols).apply(foo_func, args)

This is useful when the work done by foo_func is CPU intensive for each group.
The equivalent multiprocessing call becomes,
            df_out = mp_groupby(df_in, gb_cols, gb_func, *args, **mp_args)
where
- df_in is the input df
- df_out is the output df
- gb_cols is the list of cols in the group-by call. Must be a list. If gb_cols == [], it performs a row apply.
- args is the list of arguments of gb_func
- mp_args: multiprocessing args (# CPUs, # queues, ...)

This code does not work within a class. Must use use copy_reg. See references.

References
----------

# http://stackoverflow.com/questions/22487296/multiprocessing-in-python-sharing-large-object-e-g-pandas-dataframe-between
# http://stackoverflow.com/questions/19615560/is-there-a-good-way-to-avoid-memory-deep-copy-or-to-reduce-time-spent-in-multipr
# http://stackoverflow.com/questions/25156768/cant-pickle-type-instancemethod-using-pythons-multiprocessing-pool-apply-a
# http://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-pythons-multiprocessing-pool-ma/7309686#7309686
# http://pymotw.com/2/multiprocessing/index.html
# http://pymotw.com/2/multiprocessing/basics.html
# https://www.frozentux.net/2010/05/python-multiprocessing/
# http://www.acnenomor.com/4401725p1/cant-pickle-when-using-pythons-multiprocessing-poolmap
"""

import multiprocessing as mp
import numpy as np
import pandas as pd
from operator import itemgetter
import sys


def mp_groupby(df_in, gb_cols, gb_func, *gb_func_args, **mp_args):
    """
    MP implementation of df_out = df_in.groupby(gb_cols).apply(gb_func, gb_func_args)
    :param gb_cols: list of group_by cols
    :param gb_func: function to apply on the group_by
    :param gb_func_args: gb_func args
    mp_args
    :param n_cpus: number of CPUs to use. If 0 do not use multiprocessing.
    :param n_queues: Number of queues with tasks. If n_queues != 1, it is set to n_cpus. Default is 1.
    :return: Same as non-multiprocessing group_by.

    Processing options:
    - n_queues !=1: One separate task queue per CPU. The task queue contains multiple DFs, one DF per gb combination.
    - n_queues = 1: One shared single task list. The task list contains multiple DFs, one DF per group_by combination.
    """

    # MP parameters
    # number of CPUs
    n_cpus = n_cpus_input(**mp_args)
    if n_cpus == 0:    # run without multiprocessing
        return df_in.groupby(gb_cols).apply(gb_func, *gb_func_args)
    else:              # keep proposed value but ensure it is not above the available CPUs
        n_cpus = min(n_cpus, mp.cpu_count())
    qs = mp_args['n_queues'] if 'n_queues' in mp_args else 1
    n_queues = n_cpus if qs != 1 else 1

    # build the processing groups
    if n_queues == n_cpus:                                       # create as many task queues as CPUs
        df_groups = df_grouper(df_in, gb_cols, n_cpus)           # processing groups for dfs
    else:                                                        # create a single task queue
        df_groups = df_grouper(df_in, gb_cols, 1)

    gb_func_args += (df_in, )                                           # append df_in to func args
    mp_func_by_groups = simple_parallel(func_by_groups, n_cpus)         # a general template that groups a function used in a pd groupby
    result = mp_func_by_groups(df_groups.keys(), df_groups, gb_func, *gb_func_args)
    if len(result) > 0:
        return pd.concat(result)
    else:
        return pd.DataFrame()


def n_cpus_input(**mp_args):
    if 'n_cpus' not in mp_args:
        'Missing number of CPUs'
        sys.exit(0)
    else:
        n_cpus = mp_args['n_cpus']

    if isinstance(n_cpus, int) is False:
        print 'invalid number of CPUs: ' + str(n_cpus)
        sys.exit(0)
    return n_cpus


def df_grouper(df, gb_cols, n_groups):
    """
    df: df to process.
    gb_cols: cols to perform the group_by. If gb_cols = [], there is no group_by, ie it directly performs df.apply(..., axis=1)
    n_groups: number of groups to use.
    Returns: a dict with n_groups keys 0, .., n_groups - 1 with:
             - dict[group_nbr] = [...,(start_df, len_df), ...]
             - start_df: first index in one of the DFs in the group and len_df: LENGTH of the df.
    """
    # prepare the index dict to iterate
    idx_dict = dict()
    if len(gb_cols) > 0:                                          # traditional group_by + apply
        srt_df = df.sort(columns=gb_cols).reset_index(drop=True)  # sorting by gb cols makes the index continuous in each group! All we have to do is identify group boundaries!
        g = srt_df.groupby(gb_cols)
        idx_dict = {np.min(v): len(v) for v in g.indices.values()}
    else:                                                         # plain apply (to rows?)
        df.reset_index(inplace=True, drop=True)
        sz, r = divmod(len(df), n_groups)
        start = sz + r
        idx_dict[0] = start
        for _ in range(1, n_groups):
            idx_dict[start] = sz
            start += sz

    groups = mp_balancer(idx_dict.keys(), n_groups, idx_dict.values())    # groups[k] = [tid1, tid2, ...]
    df_grp = {g: [(s, idx_dict[s]) for s in groups[g]] for g in groups}   # df_grp[k] = [..., (idx_start, len), ...]
    return df_grp


def func_by_groups(key, group_dict, func, *args):
    """
    Implements the group_by version of function func
    :param key: group key
    :param group_dict: group_dict[key] = [..., (df_start, df_len), ...]
    :param func: function to execute by group
    :param df: DF arg for func
    :param args: args for func, excluding df
    :return: DF concat of the func applied to each groupby in group_dict[key]
    """
    # d_out = pd.DataFrame()
    df = args[-1]
    args = args[:-1]                          # create a new args dropping the df
    d_list = []
    for df_start, df_len in group_dict[key]:
        d = func(df.iloc[df_start:(df_start + df_len)], *args)
        if len(d) > 0:
            d_list.append(d)
    if len(d_list) > 0:
        return pd.concat(d_list)
    else:
        return pd.DataFrame()


def mp_balancer(task_list, n_groups, weights=None):
    """
    Balance the task_list (list of task keys or task ids or task indices) into n_groups so that the amount of work is about the same in each list.
    task_list:
    weights: is a list of task weights to guide the balancing. The larger the weight the more processing needed.
    n_groups: how many processing groups to return
    return: dictionary with n_groups keys. groups[k] = task list for group k
    """
    if weights is None:
        weights = [1] * len(task_list)

    t_list = zip(task_list, weights)                          # (tid, w_tid) list
    s_list = sorted(t_list, key=itemgetter(1), reverse=True)  # sort in descending weight  [(tid, weight), ...]
    groups = {k: [] for k in range(n_groups)}                 # groups[k] = [(cid1, w1), (cid2, w2), ...]  list of tasks in group k
    w_dict = {k: 0 for k in range(n_groups)}                  # cumulative weight of every group

    next_g = 0
    for s in s_list:
        groups[next_g].append(s[0])                              # s = (tid, weight)
        w_dict[next_g] += s[1]
        next_g = min(w_dict.iteritems(), key=itemgetter(1))[0]   # assign the highest available tid to the group with lowest cummulative weight
    return groups   # groups[k] = [tid1, tid2, ...]


def simple_parallel(function, n_procs=None):
    """
    http://valentinoetal.wordpress.com/2014/06/10/stupid-parallel-pseudo-decorator-in-python/
    Works similar to a decorator to parallelize "blatantly parallel" problems.
    Decorators and multiprocessing don't play nicely because of naming issues.

    Inputs
    ======
    function : the function that will be parallelized. The FIRST
        argument must the one to be iterated on (in parallel). The other
        arguments are the same in all the parallel runs of the function.
    n_procs : int, the number of processes to run. Default is None.
        It is passed to multiprocessing.Pool (see that for details).

    Output
    ======
    A parallelized function. DO NOT NAME IT THE SAME AS THE INPUT FUNCTION.

    Example
    =======
    def _square_and_offset(value, offset=0):
        return value**2 + offset

    parallel_square_and_offset = simple_parallel(_square_and_offset,n_procs=5)
    print parallel_square_and_offset(range(10), offset=3)
    > [3, 4, 7, 12, 19, 28, 39, 52, 67, 84]
    """

    def apply(iterable_values, *args, **kwargs):
        args = list(args)
        p = mp.Pool(n_procs)
        result = [p.apply_async(function, args=[value]+args, kwds=kwargs) for value in iterable_values]
        p.close()                          # do not use p.join as it serializes the processes!
        try:
            return [r.get() for r in result]
        except KeyError:
            return []
    return apply
