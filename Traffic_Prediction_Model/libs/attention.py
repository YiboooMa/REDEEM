import torch
import numpy as np
import torch.nn.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'


###################################################################################################3

def attention_extractor(near, input):  # near.shape = k_nearest, 21   # input.shape = 1, 21
    relate = []
    time_nearest = 1
    if near.shape[0] == 0:
        near = input.resize(1, 21)
    for i in range(21):
        s = near[:, time_nearest - 1:i + time_nearest]
        key = input[time_nearest - 1:i + time_nearest]
        key = key.unsqueeze(0)  # 1,1,21
        if 1:
            s_norm = torch.norm(s, dim=1, keepdim=True)
            s_norm = torch.sqrt(s_norm * s_norm + 1e-5)
            s = s / s_norm
            key_norm = torch.norm(key, dim=1, keepdim=True)
            key_norm = torch.sqrt(key_norm * key_norm + 1e-5)
            key = key / key_norm
        score = torch.bmm(s.unsqueeze(0), torch.transpose(key, 0, 1).unsqueeze(0)).squeeze(0)
        pos = F.softmax(score, dim=0)
        neg = F.softmin(score, dim=0)
        p = torch.bmm(torch.transpose(near[:, i:i + time_nearest], 1, 0).unsqueeze(0),
                      pos.unsqueeze(0)).squeeze(0)
        n = torch.bmm(torch.transpose(near[:, i:i + time_nearest], 1, 0).unsqueeze(0),
                      neg.unsqueeze(0)).squeeze(0)
        relate.append(torch.cat((p, n), dim=0).squeeze(1).unsqueeze(0))

    return np.concatenate((torch.cat(relate, dim=0).numpy(), input.numpy().reshape(21, 1)), axis=1)  # 21, 2*time_nearest+1

