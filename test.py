#
# import torch
# import numpy as np
# import torch.nn as nn
# batch_size = 10
#
# one = torch.Tensor(np.ones((batch_size, 3, 96, 96)))
# zero = torch.Tensor(np.zeros((batch_size, 3, 96, 96)))
# MSE_criterion = nn.MSELoss(reduction='sum')/batch_size
#
# print(MSE_criterion(one, zero))

from tqdm import tqdm
pbar = tqdm(["a", "b", "c", "d"])
num_vowels = 0
for ichar in pbar:
    if ichar in ['a','e','i','o','u']:
        num_vowels += 1
    pbar.set_postfix({'num_vowels': num_vowels})