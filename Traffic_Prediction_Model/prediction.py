from libs.attention import attention_extractor
from libs.data_split import data_generate
from libs.causality_test import neighbors_select_by_causality
from model.TCN import TCN
from model.mlp import MLP
from tqdm import tqdm
import os
import torch
import copy
import sys
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np



def Shuffle(x, y, need_to_tensor=True):
    if type(x) == torch.tensor:
        x = x.cpu().numpy()
    if type(y) == torch.tensor:
        y = y.cpu().numpy()
    idx = np.array(range(len(x)))
    np.random.shuffle(idx)
    x = x[idx.tolist()]
    y = y[idx.tolist()]
    if need_to_tensor:
        x = torch.tensor(x.cpu().detach().numpy()).to(torch.float32).cuda()
    if need_to_tensor:
        y = torch.tensor(y.cpu().detach().numpy()).to(torch.float32).cuda()
    return x, y


def data_after_attention(Data_dict, Neighbors_AfterCausal, n_train, n_valid, n_test, n_pred, if_shuffle=False):
    Data_dict_neighbor_list = []
    Input_dict = {'X': {}, 'Y': Data_dict['Y']}
    for i in range(Neighbors_AfterCausal.shape[0]):
        Data_dict_neighbor = data_generate(Neighbors_AfterCausal[i], if_shuffle)
        Data_dict_neighbor_list.append(Data_dict_neighbor)

    X_train_AfterAttention = []
    for j in range(n_train):
        Train_oneneighbor_list = []
        for i in range(Neighbors_AfterCausal.shape[0]):
            Train_oneneighbor_list.append(Data_dict_neighbor_list[i]['X']['train'][j])
        Train_oneneighbor_list = np.array(Train_oneneighbor_list)
        X_train_AfterAttention.append(
            attention_extractor(torch.tensor(Train_oneneighbor_list), torch.tensor(Data_dict['X']['train'][j])))
    X_train_AfterAttention = np.array(X_train_AfterAttention)
    Input_dict['X']['train'] = X_train_AfterAttention

    X_valid_AfterAttention = []
    for j in range(n_valid):
        Valid_oneneighbor_list = []
        for i in range(Neighbors_AfterCausal.shape[0]):
            Valid_oneneighbor_list.append(Data_dict_neighbor_list[i]['X']['valid'][j])
        Valid_oneneighbor_list = np.array(Valid_oneneighbor_list)
        X_valid_AfterAttention.append(
            attention_extractor(torch.tensor(Valid_oneneighbor_list), torch.tensor(Data_dict['X']['valid'][j])))
    X_valid_AfterAttention = np.array(X_valid_AfterAttention)
    Input_dict['X']['valid'] = X_valid_AfterAttention

    X_test_AfterAttention = []
    for j in range(n_test):
        Test_oneneighbor_list = []
        for i in range(Neighbors_AfterCausal.shape[0]):
            Test_oneneighbor_list.append(Data_dict_neighbor_list[i]['X']['test'][j])
        Test_oneneighbor_list = np.array(Test_oneneighbor_list)
        X_test_AfterAttention.append(
            attention_extractor(torch.tensor(Test_oneneighbor_list), torch.tensor(Data_dict['X']['test'][j])))
    X_test_AfterAttention = np.array(X_test_AfterAttention)
    Input_dict['X']['test'] = X_test_AfterAttention

    X_pred_AfterAttention = []
    for j in range(n_pred):
        Pred_oneneighbor_list = []
        for i in range(Neighbors_AfterCausal.shape[0]):
            Pred_oneneighbor_list.append(Data_dict_neighbor_list[i]['X']['pred'][j])
        Pred_oneneighbor_list = np.array(Pred_oneneighbor_list)
        X_pred_AfterAttention.append(
            attention_extractor(torch.tensor(Pred_oneneighbor_list), torch.tensor(Data_dict['X']['pred'][j])))
    X_pred_AfterAttention = np.array(X_pred_AfterAttention)
    Input_dict['X']['pred'] = X_pred_AfterAttention

    return Input_dict


def dataset_generator(Cell_list, Traffic_dict, Traffic_all_cell, if_shuffle=False, cell_need=''):
    Cell_list_used = Cell_list
    Traffic_used = Traffic_dict
    cell = Cell_list_used[0]
    if cell_need:
        cell = cell_need
    Data = np.array(Traffic_used[cell])
    Data_dict = data_generate(Data, if_shuffle)

    Cell_neighbors = Cell_equivalent_list[cell]
    Neighbors_AfterCausal = neighbors_select_by_causality(Traffic_all_cell, Cell_neighbors, cell)

    n_train = Data_dict['X']['train'].shape[0]
    n_valid = Data_dict['X']['valid'].shape[0]
    n_test = Data_dict['X']['test'].shape[0]
    n_pred = Data_dict['X']['pred'].shape[0]
    Input_dict = data_after_attention(Data_dict, Neighbors_AfterCausal, n_train, n_valid, n_test, n_pred, if_shuffle=if_shuffle)
    Input_dict['mean'] = Data_dict['mean']
    Input_dict['std'] = Data_dict['std']

    return Input_dict



def final_model_train(Input_dict, TCNModel, num_blocks=7, epoch_max = 100):
    
    drop_out = 0.1
    batch_size = 64
    learning_rate = 0.001

    n_train = len(Input_dict['X']['train'])
    n_valid = len(Input_dict['X']['valid'])
    n_test = len(Input_dict['X']['test'])

    x_train = torch.tensor(Input_dict['X']['train']).to(torch.float32).cuda()
    # [n_val,21,3]
    x_valid = torch.tensor(Input_dict['X']['valid']).to(torch.float32).cuda()
    # [n_test,21,3]
    x_test = torch.tensor(Input_dict['X']['test']).to(torch.float32).cuda()

    # [N,1]
    y_train = torch.tensor(Input_dict['Y']['train']).to(torch.float32).cuda()
    y_valid = torch.tensor(Input_dict['Y']['valid']).to(torch.float32).cuda()
    y_test = torch.tensor(Input_dict['Y']['test']).to(torch.float32).cuda()

    model_params = {
        # 'input_size',C_in
        'input_size': 21,
        'output_size': 21,
        'num_channels': [50] * 4,
        'kernel_size': 3,
        'dropout': drop_out
    }

    Model = nn.Sequential()
    for i in range(num_blocks):
        Model.add_module(f'TCN Block {i+1}', TCNModel(**model_params).cuda())
    Model.add_module('MLP', MLP(input_size = 21 * 3, hidden_size=7).cuda())

    TCN_loss = nn.MSELoss()

    best_params = None
    min_val_loss = sys.maxsize

    training_loss = []
    validation_loss = []
    testing_loss = []
    n_input = 0
    for epoch in range(epoch_max):
        for batch_idx in range(n_train // batch_size + 1):

            x_train_, y_train_ = Shuffle(
                x_train[batch_idx * batch_size:np.min([(batch_idx + 1) * batch_size, n_train])],
                y_train[batch_idx * batch_size:np.min([(batch_idx + 1) * batch_size, n_train])],
                need_to_tensor=True)
            output = Model(x_train_.float())
            loss = TCN_loss(output, y_train_)

            optimizer = torch.optim.Adam(params=Model.parameters(), lr=learning_rate)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            val_prediction = Model(x_valid)
            val_loss = TCN_loss(val_prediction, y_valid)

            test_prediction = Model(x_test)
            test_loss = TCN_loss(test_prediction, y_test)

            training_loss.append(loss.item())
            validation_loss.append(val_loss.item())
            testing_loss.append(test_loss.item())

            if val_loss.item() < min_val_loss:
                best_params = copy.deepcopy(Model.state_dict())
                min_val_loss = val_loss.item()

            n_input += 1

    return Model



# def use_gpu(used_percentage=0.75):
#     nvmlInit()
#     gpu_num = nvmlDeviceGetCount()
#     out = ""
#     for i in range(gpu_num):
#         handle = nvmlDeviceGetHandleByIndex(i)
#         info = nvmlDeviceGetMemoryInfo(handle)
#         used_percentage_real = info.used / info.total
#         if out == "":
#             if used_percentage_real < used_percentage:
#                 out += str(i)
#         else:
#             if used_percentage_real < used_percentage:
#                 out += "," + str(i)
#     nvmlShutdown()
#     return out


######################## Generate the input #############################

net = '5G'
if_shuffle = True

Traffic_4G = np.load(r'data/Traffic_4G.npy', allow_pickle=True).item()
Traffic_5G = np.load(r'data/Traffic_5G.npy', allow_pickle=True).item()
Cell_equivalent_list = np.load(r'data/Cell_equlist.npy', allow_pickle=True).item()
Cell_List_4G = list(Traffic_4G.keys())
Cell_List_5G = list(Traffic_5G.keys())
Traffic_all_dict = {'4G': Traffic_4G, '5G': Traffic_5G}
Traffic_all_cell = dict(Traffic_4G, **Traffic_5G)
Cell_List_all_dict = {'4G': Cell_List_4G, '5G': Cell_List_5G}
Cell_List_all_cell = Cell_List_4G + Cell_List_5G


Traffic_Input = {}
print(f'Number of {net} cell: {len(Cell_List_all_dict[net])}')
for cell in tqdm(Cell_List_all_dict[net]):
    Input_dict = dataset_generator(Cell_list=Cell_List_all_dict[net], Traffic_dict=Traffic_all_dict[net],
                                    Traffic_all_cell=Traffic_all_cell,
                                    if_shuffle=if_shuffle, cell_need=cell)
    Traffic_Input[cell] = Input_dict
np.save(f'data/Traffic_Input_{net}.npy', Traffic_Input)


######################## Train the model and get the prediction #############################
num_blocks = 7
epoch_max = 10
# os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu(0.85)  # Choose device
Traffic = np.load(f'data/Traffic_{net}.npy', allow_pickle=True).item()
T_pred = {}
for cell in Traffic_Input:
    Model = final_model_train(Traffic_Input[cell], TCN, num_blocks, epoch_max)
    x_input = torch.tensor(Traffic_Input[cell]['X']['pred'])

    prediction = np.zeros(336)
    prediction[:21] = Traffic[cell][:21]
    for time in range(45):
        x = x_input[time*7].to(torch.float32).cuda().reshape(1, 21, 3)
        x = Model(x)
        prediction[21+time*7:28+time*7] = x.cpu().detach().numpy()
    prediction[21:] = prediction[21:] * Traffic_Input[cell]['std'] + Traffic_Input[cell]['mean']
    T_pred[cell] = prediction
np.save(f'data/Traffic_pred_{net}.npy', T_pred)
