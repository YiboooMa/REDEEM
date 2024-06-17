import numpy as np
from Power_RRU_Model import Cell_Power

def Mesh_Power_RRU(dict_onoff, Cell_Info, net):
    stu = ['Sleep', 'Active']
    for day in dict_onoff:
        for ts in dict_onoff[day]:
            for mesh in dict_onoff[day][ts]:
                for cell in dict_onoff[day][ts][mesh]:
                    if cell in Cell_Info[net]:
                        dict_onoff[day][ts][mesh][cell]['Power_RRU'] = Cell_Power(cell,
                                stu[dict_onoff[day][ts][mesh][cell]['Status']],
                                dict_onoff[day][ts][mesh][cell]['Traffic'],
                                Cell_Info)
    return dict_onoff


def ActiveBS(onoff, BSandCell):
    onoff_BS = {}
    for BS in BSandCell:
         onoff_BS[BS] = np.zeros(336)
    for day in onoff:
        for ts in onoff[day]:
            for mesh in onoff[day][ts]:
                for cell in onoff[day][ts][mesh]:
                    if dict_onoff[day][ts][mesh][cell]['Status'] == 1:
                        onoff_BS[cell.split('-')[2]][day*48+ts] = 1
    return onoff_BS


def Power_get(dict_PeAs, BBU, onf_BS, BSandCell):
    Pe = {}
    for bs in BBU:
        Pe[bs] = BBU[bs]
    for day in dict_PeAs:
        for ts in dict_PeAs[day]:
            for mesh in dict_PeAs[day][ts]:
                for cell in dict_PeAs[day][ts][mesh]:
                    bs = cell.split('-')[2]
                    if cell in BSandCell[bs]:
                        try:
                            Pe[bs][day*48+ts] += dict_PeAs[day][ts][mesh][cell]['Power_RRU']
                        except:
                            print(bs, cell, bs in Pe)
    for bs in Pe:
        Pe[bs] = Pe[bs] * onf_BS[bs]
    return Pe

net = '5G'
dict_onoff = np.load(f'data/Onoff_Traffic_arrange_TCN_{net}.npy', allow_pickle=True).item()
Cell_Info = np.load(f'data/Cell_Info.npy', allow_pickle=True).item()
Traffic = np.load(f'Traffic_Prediction_Model/data/Traffic_{net}.npy', allow_pickle=True).item()
BSandCell = {}
for cell in Traffic:
    bs = cell.split('-')[2]
    if bs not in BSandCell:
        BSandCell[bs] = []
    BSandCell[bs].append(cell)
BBU = np.load(f'data/BBU_{net}.npy', allow_pickle=True).item()
onoff_BS = ActiveBS(dict_onoff, BSandCell)
dict_onoff_PeAs = Mesh_Power_RRU(dict_onoff, Cell_Info, net)
Pe = Power_get(dict_onoff_PeAs, BBU, onoff_BS, BSandCell)

EE_Mesh = {}
for day in dict_onoff_PeAs:
    EE_Mesh[day] = {}
    for ts in dict_onoff_PeAs[day]:
        EE_Mesh[day][ts] = {}
        for mesh in dict_onoff_PeAs[day][ts]:
            traffic = 0
            power = 0
            for cell in dict_onoff_PeAs[day][ts][mesh]:
                traffic += dict_onoff_PeAs[day][ts][mesh][cell]['Traffic']
                power += dict_onoff_PeAs[day][ts][mesh][cell]['Power_RRU']
            if traffic == 0:
                EE_Mesh[day][ts][mesh] = 0
            else:
                assert power > 0
                EE_Mesh[day][ts][mesh] = traffic/power

np.save(f'data/Onoff_{net}BS_TCN.npy', onoff_BS)
np.save(f'data/Onoff_Power_arrange_TCN_{net}.npy', dict_onoff_PeAs)
np.save(f'data/Pe_AfterTCN_{net}.npy', Pe)
np.save(f'data/Mesh_Energy_Efficiency_{net}.npy', EE_Mesh)