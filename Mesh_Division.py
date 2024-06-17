from geopy.distance import geodesic

LoLa_5G = {}

for cell in Cell_Info:
    LoLa_5G[cell] = {'longitude':Info_5G_old[cell]['经度'], 'latitude':Info_5G_old[cell]['纬度']}

def calculate_distance(coord1, coord2):
    distance = geodesic(coord1, coord2).meters
    return distance

Cell_equ = {}
print('等效小区搜索')
for cell in tqdm(LoLa_5G):
    Cell_equ[cell] = []
    for cell_ in LoLa_5G:
        if cell.split('-')[2] != cell_.split('-')[2]:
            distance = calculate_distance((LoLa_5G[cell]['latitude'], LoLa_5G[cell]['longitude']), 
                                          (LoLa_5G[cell_]['latitude'], LoLa_5G[cell_]['longitude']))
            if distance < 300:
                Cell_equ[cell].append(cell_)

print('网格划分')
Grid = {}
idx = 0
cell_divided = {}
for cell in tqdm(Cell_equ):
    if cell not in cell_divided:
        idx += 1
        Grid[idx] = [cell]
        cell_divided[cell] = 0
        for cellx in Cell_equ[cell]:
            if cellx not in cell_divided:
                equ = True
                for celly in Grid[idx]:
                    if celly not in Cell_equ[cellx]:
                        equ = False
                if equ:
                    Grid[idx].append(cellx)
                    cell_divided[cellx] = 0
len(Grid)