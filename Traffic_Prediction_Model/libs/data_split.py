import numpy as np
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'


def data_norm(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    if std.any == 0:
        std = np.ones(data.shape[1])*np.max(mean)
    norm_data = (data - mean) / std
    return norm_data, mean, std


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


def get_data(price, train_period, pre_period):
    data_len = len(price)
    x = []
    y = []
    for i in range(data_len - train_period - pre_period + 1):
        x.append([a for a in price[i:i + train_period]])
        y.append(price[i + train_period:i + train_period + pre_period].tolist())
    x = np.array(x)
    y = np.array(y)
    if np.array(x).shape[1] != train_period or np.array(y).shape[1] != pre_period:
        print('get data error')
    return x, y


def split_data(price_data, train_radio, valid_radio, train_period, pre_period, flow_max, if_shuffle=False):
    norm_para = {}
    X, Y = get_data(price_data, train_period, pre_period)
    X_pred = X.copy()

    if if_shuffle:
        X_shuffle, Y_shuffle = Shuffle(X, Y, need_to_tensor=False)
        X, Y = X_shuffle, Y_shuffle

    train_num = int(X.shape[0] * train_radio)
    valid_num = int(X.shape[0] * valid_radio)
    test_num = X.shape[0] - train_num - valid_num
    train_num += test_num - valid_num

    train_x = X[:train_num]
    train_y = Y[:train_num]
    valid_x = X[train_num:train_num + valid_num]
    valid_y = Y[train_num:train_num + valid_num]
    test_x = X[train_num + valid_num:]
    test_y = Y[train_num + valid_num:]
    
    norm_para['max'] = flow_max
    return train_x, train_y, valid_x, valid_y, test_x, test_y, norm_para, X_pred


def data_generate(Data, if_shuffle=False):
    train_period = 21
    pre_period = 7
    train_ratio = 0.8
    valid_ratio = 0.1
    Data_norm, mean, std = data_norm(Data)

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test, _, X_pred = split_data(Data_norm, train_ratio, valid_ratio,
                                                                               train_period, pre_period,
                                                                               np.max(Data_norm), if_shuffle=if_shuffle)

    Data_dict = {'X': {'train': X_train, 'valid': X_valid, 'test': X_test, 'pred': X_pred}, 'Y': {'train': Y_train, 'valid': Y_valid, 'test': Y_test}}
    
    Data_dict['mean'] = mean
    Data_dict['std'] = std

    return Data_dict