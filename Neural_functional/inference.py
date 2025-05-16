import numpy as np
import torch
import os
import conv_network as net
import matplotlib.pyplot as plt
import create_training_data as gen
import pandas

model = net.conv_neural_func5()
model.load_state_dict(torch.load("decent_conv_net.pth"))
model.to(torch.device("cuda"))

parent_dir = os.path.dirname(os.getcwd())
# Path to the Density_profiles directory
density_profiles_dir =  "/home/hagen/Documents/master_project/Data_generation/Density_profiles"
tag = "parallel2025-05-09_18-33-17"
density_profiles_dir = os.path.join(density_profiles_dir, tag)





_,_,window_dims,_ =next(model.children()).weight.shape


names= [f for f in os.listdir(density_profiles_dir) if os.path.isfile(os.path.join(density_profiles_dir, f))]
df,_ = gen.load_df(names[-1])
n_windows=int(np.sqrt(len(df))/window_dims)
window,target = gen.cut_density_windows(df,window_dims,n_windows)

window = window.reshape(1,1,window_dims,window_dims)
window = torch.tensor(window,dtype=torch.float32)

