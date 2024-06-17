import numpy as np
from tqdm import tqdm

Cell_Equal_lst = np.load('data/Cell_equlist.npy', allow_pickle=True).item()

Mesh = {}
idx = 0
cell_divided = {}
for cell in tqdm(Cell_Equal_lst):
    if cell not in cell_divided:
        idx += 1
        Mesh[idx] = [cell]
        cell_divided[cell] = 0
        for cellx in Cell_Equal_lst[cell]:
            if cellx not in cell_divided:
                equ = True
                for celly in Mesh[idx]:
                    if celly not in Cell_Equal_lst[cellx]:
                        equ = False
                if equ:
                    Mesh[idx].append(cellx)
                    cell_divided[cellx] = 0
                    
np.save('data/Mesh.npy', Mesh)