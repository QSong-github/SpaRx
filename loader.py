import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch
import scanpypip.preprocessing as pp
import scanpy as sc
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_planetoid_data
from torch_geometric.data import Data

import utils.helper as ut


def split_data(all_index, val=0.2, random_state=42):
    idx_train, idx_test = train_test_split(range(all_index), test_size=val, random_state=random_state)
    return idx_train, idx_test

def transfer_data(idx, csv, bulk_label, adj):
    cell_dict = {}
    index_names = csv.index.tolist()
    for i in idx:
        name = index_names[i]
        cell_dict[name] = i
    # adj['first'] = adj['cell1'].map(cell_dict)
    # adj['second'] = adj['cell2'].map(cell_dict)
    # print(adj['first'], adj['second'])
    # exit(0)
    e0 = []
    e1 = []
    for i1, i2 in zip(adj['cell1'].tolist(), adj['cell2'].tolist()):
        if i1 in cell_dict and i2 in cell_dict:
            # print(i1, i2, cell_dict[i1], cell_dict[i2])
            e0.append(cell_dict[i1])
            e1.append(cell_dict[i2])
    
    e0 = np.array(e0)-1
    e1 = np.array(e1)-1

    edgeList = np.array((e0, e1))
    X = csv.values
    Y = bulk_label['label_cat']
    data = Data(edge_index=torch.LongTensor(np.array(
        [edgeList[0], edgeList[1]])), x=torch.FloatTensor(X), y=torch.FloatTensor(Y))
    return data

def load_data(root='', adj_root='', label_root='',num_val=0.2, random_state=42):
    csv = pd.read_csv(root, header=0, index_col=0)
    adj = pd.read_csv(adj_root, header=0, index_col=0)
    bulk_label = pd.read_csv(label_root, header=0, index_col=0)
    bulk_label['label'] = bulk_label['label'].astype('category')
    codes = bulk_label['label'].cat.codes
    bulk_label['label_cat'] = bulk_label['label'].cat.codes
    
    train_idx, test_idx = split_data(csv.shape[0], val=num_val, random_state=random_state)
    
    cell_dict = dict(zip(csv.index, range(csv.shape[0])))
    adj['first'] = adj['cell1'].map(cell_dict)
    adj['second'] = adj['cell2'].map(cell_dict)
    e0 = adj['first'].to_numpy()
    e1 = adj['second'].to_numpy()
    edgeList = np.array((e0, e1))

    X = csv.values
    Y = bulk_label['label_cat'].values
    data = Data(edge_index=torch.LongTensor(np.array(
        [edgeList[0], edgeList[1]])), x=torch.FloatTensor(X), y=torch.LongTensor(Y))
    data.train_mask = train_idx

    # cells 2000 

    return data, train_idx, test_idx

def load_unsuper_data(root='', adj_root='', label_root='', return_index=False):
    # label_root only used for validation
    csv = pd.read_csv(root, header=0, index_col=0)
    adj = pd.read_csv(adj_root, header=0, index_col=0)
    bulk_label = pd.read_csv(label_root, header=0, index_col=0)
    bulk_label['label'] = bulk_label['label'].astype('category')
    codes = bulk_label['label'].cat.codes
    bulk_label['label_cat'] = bulk_label['label'].cat.codes

    # tranfer data
    cell_dict = dict(zip(csv.index, range(csv.shape[0])))
    adj['first'] = adj['cell1'].map(cell_dict)
    adj['second'] = adj['cell2'].map(cell_dict)
    e0 = adj['first'].to_numpy()
    e1 = adj['second'].to_numpy()
    edgeList = np.array((e0, e1))

    X = csv.values
    Y = bulk_label['label_cat'].values
    data = Data(edge_index=torch.LongTensor(np.array(
        [edgeList[0], edgeList[1]])), x=torch.FloatTensor(X), y=torch.LongTensor(Y))
    # target = torch.LongTensor(Y)
    if not return_index:
        return data
    else:
        return data, csv.index