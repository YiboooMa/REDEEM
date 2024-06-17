import numpy as np
from Power_RRU_Model import Cell_Power

def Mesh_Power_RRU(dict_onoff, Cell_Info):
    print('开始计算小区能耗：')
    stu = ['Sleep', 'Active']
    for day in dict_onoff:
        for ts in dict_onoff[day]:
            for mesh in dict_onoff[day][ts]:
                for cell in dict_onoff[day][ts][mesh]:
                    dict_onoff[day][ts][mesh][cell]['Power_RRU'] = Cell_Power(cell,
                            stu[dict_onoff[day][ts][mesh][cell]['Status']],
                            dict_onoff[day][ts][mesh][cell]['Traffic'],
                            Cell_Info)
        print('第{}天计算完成'.format(day))
    return dict_onoff


def ActiveBS(onoff, BSandCell):
    onoff_BS = {}
    print('开始判断基站启停')
    for BS in BSandCell:
         onoff_BS[BS] = np.zeros(336)
    for day in onoff:
        for ts in onoff[day]:
            for mesh in onoff[day][ts]:
                for cell in onoff[day][ts][mesh]:
                    if dict_onoff[day][ts][mesh][cell]['Status'] == 1:
                        onoff_BS[cell.split('-')[2]][day*48+ts] = 1
        print('第{}天判断完成'.format(day))
    return onoff_BS


def Power_get(dict_PeAs, BBU, onf_BS):
    print('开始获取基站能耗')
    Pe = {}
    for bs in BBU:
        Pe[bs] = BBU[bs]
    for day in dict_PeAs:
        for ts in dict_PeAs[day]:
            for mesh in dict_PeAs[day][ts]:
                for cell in dict_PeAs[day][ts][mesh]:
                    bs = cell.split('-')[2]
                    Pe[bs][day*48+ts] += dict_PeAs[day][ts][mesh][cell]['Power_RRU']
        print('第{}天获取完成'.format(day))
    for bs in Pe:
        Pe[bs] = Pe[bs] * onf_BS[bs]
    return Pe

net = '4G'
dict_onoff = np.load(f'data/Onoff_Traffic_arrange_TCN_{net}.npy', allow_pickle=True).item()
BSandCell = np.load(f'data/BSandCell_{net}.npy', allow_pickle=True).item()
BBU = np.load('./data/BBU_4G.npy', allow_pickle=True).item()
onoff_BS = ActiveBS(dict_onoff, BSandCell)
dict_onoff_PeAs = Mesh_Power_RRU(dict_onoff)
Pe = Power_get(dict_onoff_PeAs, BBU, onoff_BS)

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
            EE_Mesh[day][ts][mesh] = traffic/power

np.save(f'data/Onoff_{net}BS_TCN.npy', onoff_BS)
np.save(f'data/Onoff_Power_arrange_TCN_{net}.npy', dict_onoff_PeAs)
np.save(f'data/Pe_AfterTCN_{net}.npy', Pe)
np.save(f'data/Mesh_Energy_Efficiency_{net}.npy', EE_Mesh)