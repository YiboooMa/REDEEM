import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import os
os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'


def neighbors_select_by_causality(Traffic, equivalent_neighbors_list, cell):
    k_nearest = np.min([12, len(equivalent_neighbors_list)])
    cn = []
    select_result = []
    for neighbor in equivalent_neighbors_list:
        traffic_twocell_list = [np.array(Traffic[cell]).tolist(), np.array(Traffic[neighbor]).tolist()]
        traffic_twocell_array = np.array(traffic_twocell_list)
        select_df = pd.DataFrame(traffic_twocell_array.T, columns=['a', 'b'])
        try:
            gr = grangercausalitytests(select_df[['a', 'b']], maxlag=2, verbose=False)
            f1 = [gr[1][0][t][1] for t in gr[1][0]]
            f2 = [gr[2][0][t][1] for t in gr[2][0]]
            f = np.array(f1) + np.array(f2)
        except:
            # print('Granger causality test error', select_df[['a', 'b']])
            f = 0

        cn.append((np.mean(f), neighbor))
    am = sorted(cn, key=lambda x: x[0])
    index_near = [am[i][1] for i in range(k_nearest)]
    for neighbor_nearest in index_near:
        select_result.append(Traffic[neighbor_nearest])
    select_result = np.array(select_result)

    return select_result