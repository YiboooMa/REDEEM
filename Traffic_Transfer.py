import numpy as np

net = '4G'
Cell_Info = np.load(f'data/Cell_Info.npy', allow_pickle=True).item()
Traffic = np.load(f'data/Traffic_{net}.npy', allow_pickle=True).item()
Traffic_pred = np.load(f'Traffic_Prediction_Model/data/Traffic_pred_{net}.npy', allow_pickle=True).item()
Mesh = np.load(f'data/Mesh_{net}.npy', allow_pickle=True).item()
EnergyEfficiency = np.load(f'data/Cell_Energy_Efficiency_{net}.npy', allow_pickle=True).item()

def cell_arrange(Cell_Info, Mesh, EE, net):
    dict_arr = {}
    for i in Mesh:
        list_cell = Mesh[i].copy()
        dict_arr[i] = {}
        list_cell_arr = []
        list_capa_arr = []
        while list_cell != []:
            max = 0
            cell_max = ''
            for x in list_cell:
                if x in Cell_Info[net]:
                    if EE[x] > max:
                        max = EE[x]
                        cell_max = x
            list_cell_arr.append(cell_max)
            list_capa_arr.append(Cell_Info[net][cell_max]['Capacity'])
            list_cell.remove(cell_max)
        dict_arr[i]['id'] = list_cell_arr
        dict_arr[i]['Capacity'] = list_capa_arr
    return dict_arr

def onoff(dict_cell, dict_EV, T):
    dict_onoff = {}
    for day in range(7):
        dict1 = {}
        for ts in range(48):
            dict2 = {}
            for mesh in dict_cell:
                dict3 = {}
                C_sum = 0
                for i in range(len(dict_cell[mesh]['id'])):
                    cell = dict_cell[mesh]['id'][i]
                    dict4 = {}
                    if C_sum < dict_EV[day][mesh][ts]:
                        C_sum += dict_cell[mesh]['Capacity'][i]
                        dict4['Status'] = 1
                    else:
                        dict4['Status'] = 0
                    dict4['Capacity'] = dict_cell[mesh]['Capacity'][i]
                    dict3[cell] = dict4
                dict2[mesh] = dict3
            dict1[ts] = dict2
        dict_onoff[day] = dict1

    CS2 = {}
    for g in Mesh:
        for cell in Mesh[g]:
            CS2[cell] = []
            for day in dict_onoff:
                for ts in dict_onoff[day]:
                    CS2[cell].append(C[cell]['Capacity'] * dict_onoff[day][ts][g][cell]['Status'])

    for day in dict_onoff:
        for ts in dict_onoff[day]:
            for mesh in dict_onoff[day][ts]:
                Tm = 0
                for cell in dict_onoff[day][ts][mesh]:
                    if dict_onoff[day][ts][mesh][cell]['Status'] == 0:
                        Tm += T[cell][day*48+ts]
                        dict_onoff[day][ts][mesh][cell]['Traffic'] = 0
                for i in range(len(dict_cell[mesh]['id'])):
                    cell = dict_cell[mesh]['id'][i]
                    if dict_onoff[day][ts][mesh][cell]['Status'] == 1:
                        if T[cell][day*48+ts] + Tm > dict_onoff[day][ts][mesh][cell]['Capacity']:
                            dict_onoff[day][ts][mesh][cell]['Traffic'] = dict_onoff[day][ts][mesh][cell]['Capacity']
                            Tm = Tm - (dict_onoff[day][ts][mesh][cell]['Capacity'] - T[cell][day*48+ts])
                        else:
                            dict_onoff[day][ts][mesh][cell]['Traffic'] = T[cell][day*48+ts] + Tm
                            Tm = 0
                        if Tm < 0 or dict_onoff[day][ts][mesh][cell]['Traffic'] < 0:
                            raise Exception(f"Traffic Transfer Error! {cell}, {Tm}, {dict_onoff[day][ts][mesh][cell]['Traffic']}")
    return CS2, dict_onoff

dict_arr = cell_arrange(Cell_Info, Mesh, EnergyEfficiency)
TT = {}
for day in range(7):
    TT[day] = {}
    for mesh in Mesh:
        TT[day][mesh] = {}
        for ts in range(48):
            TT[day][mesh][ts] = 0
            for cell in Mesh[mesh]:
                TT[day][mesh][ts] += Traffic_pred[cell][day*48+ts]
CS2, dict_onoff = onoff(dict_arr, TT, Traffic)
np.save(f'data/Onoff_Traffic_arrange_TCN_{net}.npy', dict_onoff)