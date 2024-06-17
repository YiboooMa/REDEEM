import numpy as np
from Power_RRU_Model import Cell_Power

Mesh = np.load('data/Mesh.npy', allow_pickle=True).item()
Cell_Equal = np.load('data/Cell_equlist.npy', allow_pickle=True).item()
Cell_Info = np.load(f'data/Cell_Info.npy', allow_pickle=True).item()
BBU_4G = np.load('./data/BBU_4G.npy', allow_pickle=True).item()
BBU_5G = np.load('./data/BBU_5G.npy', allow_pickle=True).item()
BBU = dict(BBU_4G, **BBU_5G)
BSandCell_4G = {}
BSandCell_5G = {}
for cell in Cell_Info['4G']:
    bs = cell.split('-')[2]
    if bs not in BSandCell_4G:
        BSandCell_4G[bs] = []
    BSandCell_4G[bs].append(cell)
for cell in Cell_Info['5G']:
    bs = cell.split('-')[2]
    if bs not in BSandCell_5G:
        BSandCell_5G[bs] = []
    BSandCell_5G[bs].append(cell)
Cell_Mesh = {}
for c in Cell_Info['4G']:
    for g in Mesh:
        if c in Mesh[g]:
            Cell_Mesh[c] = g


def Mesh_Power_RRU(dict_onoff, net):
    stu = ['Sleep', 'Active']
    if net == '4G':
        for day in dict_onoff:
            for ts in dict_onoff[day]:
                for mesh in dict_onoff[day][ts]:
                    for cell in dict_onoff[day][ts][mesh]:
                        if dict_onoff[day][ts][mesh][cell]['Traffic'] > Cell_Info[net][cell]['Capacity']:
                            raise Exception(f'{cell} Traffic Error!')
                        po = dict_onoff[day][ts][mesh][cell]['Power_RRU']
                        dict_onoff[day][ts][mesh][cell]['Power_RRU'] = Cell_Power(cell,
                                stu[dict_onoff[day][ts][mesh][cell]['Status']],
                                dict_onoff[day][ts][mesh][cell]['Traffic'],
                                Cell_Info)
                        if dict_onoff[day][ts][mesh][cell]['Power_RRU'] > 4*po:
                            pn = dict_onoff[day][ts][mesh][cell]['Power_RRU']
                            raise Exception(f'{cell} Power Error!')
    else:
        for day in dict_onoff:
            for ts in dict_onoff[day]:
                for cell in dict_onoff[day][ts]:
                    dict_onoff[day][ts][cell]['Power_RRU'] = Cell_Power(cell,
                            stu[dict_onoff[day][ts][cell]['Status']],
                            dict_onoff[day][ts][cell]['Traffic'],
                            Cell_Info)
    return dict_onoff


def ActiveBS(onoff, B, net):
    onoff_BS = {}
    for BS in B:
         onoff_BS[BS] = np.zeros(336)
    if net == '4G':
        for day in onoff:
            for ts in onoff[day]:
                for mesh in onoff[day][ts]:
                    for cell in onoff[day][ts][mesh]:
                        if onoff[day][ts][mesh][cell]['Status'] == 1:
                            onoff_BS[cell.split('-')[2]][day*48+ts] = 1
    else:
        for day in onoff:
            for ts in onoff[day]:
                for cell in onoff[day][ts]:
                    if onoff[day][ts][cell]['Status'] == 1:
                        onoff_BS[cell.split('-')[2]][day*48+ts] = 1
    return onoff_BS


def Power_get(dict_PeAs, U, onf_BS, net):
    Pe = {}
    for bs in U:
        Pe[bs] = U[bs]
    if net == '4G':
        for day in dict_PeAs:
            for ts in dict_PeAs[day]:
                for mesh in dict_PeAs[day][ts]:
                    for cell in dict_PeAs[day][ts][mesh]:
                        bs = cell.split('-')[2]
                        Pe[bs][day*48+ts] += dict_PeAs[day][ts][mesh][cell]['Power_RRU']
    else:
        for day in dict_PeAs:
            for ts in dict_PeAs[day]:
                for cell in dict_PeAs[day][ts]:
                    bs = cell.split('-')[2]
                    Pe[bs][day*48+ts] += dict_PeAs[day][ts][cell]['Power_RRU']
    for bs in Pe:
        Pe[bs] = Pe[bs] * onf_BS[bs]
    return Pe

Onoff = np.load('data/Onoff_Power_arrange_TCN_4G.npy', allow_pickle=True).item()
Onoff_5G = np.load('data/Onoff_Traffic_arrange_TCN_5G.npy', allow_pickle=True).item()
Traffic_5G = {}
for day in Onoff_5G:
    for ts in Onoff_5G[day]:
        for mesh in Onoff_5G[day][ts]:
            for cell in Onoff_5G[day][ts][mesh]:
                if cell not in Traffic_5G:
                    Traffic_5G[cell] = np.zeros(336)
                Traffic_5G[cell][day*48+ts] = Onoff_5G[day][ts][mesh][cell]['Traffic']
EE_Mesh = np.load(f'data/Mesh_Energy_Efficiency_4G.npy', allow_pickle=True).item()

Cell5GtoMesh = {}
Onoff_5G = {}
for day in Onoff:
    Onoff_5G[day] = {}
    Cell5GtoMesh[day] = {}
    for ts in Onoff[day]:
        Onoff_5G[day][ts] = {}
        Cell5GtoMesh[day][ts] = {}
        for c5 in Traffic_5G:
            Onoff_5G[day][ts][c5] = {}
            Cell5GtoMesh[day][ts][c5] = []
            traffic_5Gneed = Traffic_5G[c5][day*48+ts]
            power_5Gneed = Cell_Power(c5, 'Active', traffic_5Gneed, Cell_Info)
            energyEfficiency_5Gneed = traffic_5Gneed / power_5Gneed
            traffic_4Gcan = 0
            mesh_equ = []
            for c4 in Cell_Equal[c5]:
                if c4 in Cell_Info['4G']:
                    mesh = Cell_Mesh[c4]
                    if EE_Mesh[day][ts][mesh] > energyEfficiency_5Gneed:
                        mesh_equ.append([mesh, EE_Mesh[day][ts][mesh]])
                        traffic_4Gcan += Cell_Info['4G'][c4]['Capacity'] - Onoff[day][ts][mesh][c4]['Traffic']
            if len(mesh_equ):
                if traffic_5Gneed > traffic_4Gcan:
                    Onoff_5G[day][ts][c5]['Traffic'] = traffic_5Gneed
                    Onoff_5G[day][ts][c5]['Status'] = 1
                else:
                    Onoff_5G[day][ts][c5]['Traffic'] = traffic_5Gneed
                    Onoff_5G[day][ts][c5]['Status'] = 0
                    mesh_equ = np.array(mesh_equ)
                    mesh_equ_sorted = mesh_equ[np.lexsort(-mesh_equ.T)]
                    for i in range(mesh_equ_sorted.shape[0]):
                        mesh = mesh_equ_sorted[i][0]
                        for c4 in Mesh[mesh]:
                            if cell in Cell_Info['4G']:
                                Cell5GtoMesh[day][ts][c5].append(mesh)
                                if Onoff_5G[day][ts][c5]['Traffic'] > (Cell_Info['4G'][c4]['Capacity'] -
                                                                    Onoff[day][ts][mesh][c4]['Traffic']):
                                    Onoff_5G[day][ts][c5]['Traffic'] -= Cell_Info['4G'][c4]['Capacity'] - \
                                                                        Onoff[day][ts][mesh][c4]['Traffic']
                                    Onoff[day][ts][mesh][c4]['Traffic'] = Cell_Info['4G'][c4]['Capacity']
                                    Onoff[day][ts][mesh][c4]['Status'] = 1
                                else:
                                    Onoff[day][ts][mesh][c4]['Traffic'] += Onoff_5G[day][ts][c5]['Traffic']
                                    Onoff[day][ts][mesh][c4]['Status'] = 1
                                    Onoff_5G[day][ts][c5]['Traffic'] = 0
                                    break
                        if Onoff_5G[day][ts][c5]['Traffic'] == 0:
                            break
                    if Onoff_5G[day][ts][c5]['Traffic'] > 0:
                        Onoff_5G[day][ts][c5]['Status'] = 1
            else:
                Onoff_5G[day][ts][c5]['Traffic'] = traffic_5Gneed
                Onoff_5G[day][ts][c5]['Status'] = 1
np.save(f'data/Onoff_5GPower_AfterOffloading5G.npy', Onoff_5G)
np.save(f'data/Onoff_4GPower_AfterOffloading5G.npy', Onoff)



Onoff_Power_4G = Mesh_Power_RRU(Onoff, '4G')
onoff_BS_4G = ActiveBS(Onoff_Power_4G, BSandCell_4G, '4G')
Pe_4G = Power_get(Onoff_Power_4G, BBU_4G, onoff_BS_4G, '4G')
Onoff_Power_5G = Mesh_Power_RRU(Onoff_5G, '5G')
onoff_BS_5G = ActiveBS(Onoff_Power_5G, BSandCell_5G, '5G')
Pe_5G = Power_get(Onoff_Power_5G, BBU_5G, onoff_BS_5G, '5G')
n = 0
for bs in onoff_BS_5G:
    if onoff_BS_5G[bs][0] == 0:
        n += 1
print(n)
np.save(f'data/Onoff_Power_arrange_AfterOffloading_4G.npy', Onoff_Power_4G)
np.save(f'data/Onoff_4GBS_AfterOffloading.npy', onoff_BS_4G)
np.save(f'data/Onoff_Power_arrange_AfterOffloading_5G.npy', Onoff_Power_5G)
np.save(f'data/Onoff_5GBS_AfterOffloading.npy', onoff_BS_5G)
np.save(f'data/Pe_AfterOffloading.npy', dict(Pe_4G, **Pe_5G))