import numpy as np

def Cell_Power(Cell_ID, Cell_Status, Traffic, Cell_Info):
    if Cell_Status == 'Active':
        if Cell_ID in Cell_Info['5G']:
            Attena = Cell_Info['5G'][Cell_ID]['Attena']
            Mode = Cell_Info['5G'][Cell_ID]['SA/NSA']
            Capacity = Cell_Info['5G'][Cell_ID]['Capacity']
            if Attena == '32TR':
                if Mode == 'SA':
                    P_0 = 389.7694737
                    delta_P = 414.2184
                else:
                    P_0 = 287.35462756
                    delta_P = 376.2593
            elif Attena == '64TR':
                if Mode == 'SA':
                    P_0 = 702.56410063
                    delta_P = 379.4423
                else:
                    P_0 = 729.73145801
                    delta_P = 266.8096
            Power_RRU = P_0 + delta_P * Traffic / Capacity

        elif Cell_ID in Cell_Info['4G']:
            Attena = Cell_Info['4G'][Cell_ID]['Attena']
            Capacity = Cell_Info['4G'][Cell_ID]['Capacity']
            if Attena == '2TR':
                P_0 = 200.47
                delta_P = 114.84

            elif Attena == '4TR':
                P_0 = 214.08
                delta_P = 194.24

            elif Attena == '8TR':
                P_0 = 233.47
                delta_P = 248.34
            Power_RRU = P_0 + delta_P * Traffic / Capacity
        else:
            print('Cell Not Found!!!')
    elif Cell_Status == 'Sleep':
        assert Traffic == 0, 'Error!!!'
        if Cell_ID in Cell_Info['5G']:
            Attena = Cell_Info['5G'][Cell_ID]['Attena']
            Mode = Cell_Info['5G'][Cell_ID]['SA/NSA']
            Capacity = Cell_Info['5G'][Cell_ID]['Capacity']
            if Attena == '32TR':
                if Mode == 'SA':
                    Power_RRU = 78.9992
                else:
                    Power_RRU = 69.4302
            elif Attena == '64TR':
                if Mode == 'SA':
                    Power_RRU = 88.4698
                else:
                    Power_RRU = 90.5637

        elif Cell_ID in Cell_Info['4G']:
            Attena = Cell_Info['4G'][Cell_ID]['Attena']
            Capacity = Cell_Info['4G'][Cell_ID]['Capacity']
            if Attena == '2TR':
                Power_RRU = 119.0310
            elif Attena == '4TR':
                Power_RRU = 127.9319
            elif Attena == '8TR':
                Power_RRU = 133.9013
        else:
            print('Cell Not Found!!!')
    elif Cell_Status == 'OFF':
        assert Traffic == 0, 'Error!!!'
        Power_RRU = 0
    else:
        print('Status not found!!!')
    return Power_RRU