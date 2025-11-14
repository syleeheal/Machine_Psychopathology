import os
from typing import List, Dict
from tqdm import trange, tqdm
import gc
import re
import pandas as pd
from pandas import DataFrame
import random

import numpy as np

from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import copy


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
    def forward(self, x):
        return self.linear(x)
            
def linear_fit_torch(X, y_onehot, test_size=0.2, num_epochs = 500, lr=0.001, device = 'cuda'):
    
    model = Linear(X.size(1), y_onehot.size(1))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    X = X.to(device)
    y_onehot = y_onehot.to(device)
    model.to(X.dtype).to(X.device)
    
    train_idx = torch.randperm(X.size(0))[:int((1-test_size)*X.size(0))]
    test_idx = torch.randperm(X.size(0))[int((1-test_size)*X.size(0)):]
    
    X_train, y_train = X[train_idx], y_onehot[train_idx]
    X_test, y_test = X[test_idx], y_onehot[test_idx]

    iterator = trange(num_epochs, desc='Cls. Loss: ', leave=False)
    best_loss = np.inf
    
    for epoch in iterator:
        
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            
            if test_loss < best_loss:
                best_loss = test_loss
                best_model = model
        
        iterator.set_description('Cls. Loss: {:.4f}'.format(test_loss.item()))
            
    best_model.eval()
    outputs = best_model(X_test) 
    
    threshold = 0.5
    y_hat = (torch.sigmoid(outputs) > threshold).float()
    
    f1_macro = f1_score(y_test.cpu().numpy(), y_hat.cpu().numpy(), average='macro')
    f1_micro = f1_score(y_test.cpu().numpy(), y_hat.cpu().numpy(), average='micro')
    print(f'F1 Macro {f1_macro}; F1 Micro {f1_micro}')
    
    # print f1 for each label
    f1_per_label = f1_score(y_test.cpu().numpy(), y_hat.cpu().numpy(), average=None)
    for i, f1 in enumerate(f1_per_label):
        print(f'F1 {i}: {f1}')
        
    return best_model

