import numpy as np 
import pandas as pd
from torch import tensor,float32,save
import os
import matplotlib.pyplot as plt
import torch

def load_df(name,density_profiles_dir):
    name = os.path.join( density_profiles_dir, name)
    df = pd.read_csv(name,delimiter = " ")
    nperiod = (list(df.columns)[-2])
    mu = (list(df.columns)[-1])
    df.drop(columns=[mu],inplace=True)
    df.drop(columns=[nperiod],inplace=True)
    nperiod = int(nperiod)
    return df,nperiod


def cut_density_windows(df, window_dim,n_windows,window_stack,center_values):
    rhomatrix = df.pivot(index='y', columns='x', values='rho').values
    mulocmatrix = df.pivot(index='y', columns='x', values='muloc').values
    for i in range(n_windows):
        for j in range(n_windows):
            window = rhomatrix[i:i+window_dim,j:j+window_dim]
            center_x_index = int(i+0.5*window_dim)
            center_y_index = int(j+0.5*window_dim)
            center = mulocmatrix[center_x_index,center_y_index]
            window_stack.append(window)
            center_values.append(center)
            
    return window_stack,center_values

def cut_density_windows_torch_unpadded(df, window_dim,n_windows):
    rhomatrix = df.pivot(index='y', columns='x', values='rho').values
    rhotensor = torch.tensor(rhomatrix, dtype=torch.float32)
    unfolded_rho = rhotensor.unfold(0, window_dim, window_dim).unfold(1, window_dim, window_dim)
    windows = unfolded_rho.contiguous().view(-1, window_dim, window_dim)
    return windows

def cut_density_windows_torch_padded(df, window_dim,n_windows):
    rhomatrix = df.pivot(index='y', columns='x', values='rho').values
    pad_size = window_dim//2
    rhotensor = torch.tensor(rhomatrix, dtype=torch.float32)
    rhotensor = rhotensor.unsqueeze(0).unsqueeze(0)
    rhotensor = rhotensor.to(device="cuda")

    rho_pad = torch.nn.functional.pad(rhotensor, (pad_size, pad_size,pad_size,pad_size), mode="circular")
    rho_pad = rho_pad.squeeze(0).squeeze(0)
    unfolded_rho = rho_pad.unfold(0, window_dim, 1).unfold(1, window_dim, 1)
    print(unfolded_rho.shape)
    windows = unfolded_rho.squeeze(0).squeeze(0)
    print(windows.shape)
    windows = unfolded_rho.contiguous().view(-1, window_dim, window_dim)
    print(windows.shape)
    #windows = windows.to(device="cpu")
    return windows

def build_training_data(df, width,L,window_stack,center_values):
    n_windows = int(L/width)
    window_dim = int(np.sqrt(len(df))/n_windows)
    window,center = cut_density_windows(df, window_dim,n_windows)
    return window_stack,center_values


def save_matrices(inputs,targets):
    inputs = tensor(inputs,dtype=float32)
    targets = tensor(targets,dtype=float32)
    save({"windows": inputs, "c1": targets}, "training_data_test.pt")

