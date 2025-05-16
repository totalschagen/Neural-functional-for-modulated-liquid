import numpy as np 
import pandas as pd
import os
import neural_helper as helper
parent_dir = os.path.dirname(os.getcwd())
# Path to the Density_profiles directory
density_profiles_dir =  "/home/hagen/Documents/master_project/Data_generation/Density_profiles"
tag = "parallel2025-05-09_18-33-17"
density_profiles_dir = os.path.join(density_profiles_dir, tag)



names= [f for f in os.listdir(density_profiles_dir) if os.path.isfile(os.path.join(density_profiles_dir, f))]

window_stack = []
value_stack = []

# for i in names:
#     df,nperiod = helper.load_df(i,density_profiles_dir)
#     # Get the number of rows and columns in the DataFrame
#     df["muloc"]=np.log(df["rho"])+df["muloc"]
#     if np.sum(df["rho"])< 0.1:
#         continue
#     window_stack,value_stack= helper.cut_density_windows(df, 2.5, 15,window_stack,value_stack)
# window_stack = np.stack(window_stack)
# value_stack = np.stack(value_stack)
# print(window_stack.shape, value_stack.shape)


# helper.save_matrices(window_stack,value_stack)

df,nperiod = helper.load_df(names[0],density_profiles_dir)
n_windows = int(15/2.5)
window_dim = int(np.sqrt(len(df))/n_windows)
window_classic = helper.cut_density_windows(df, window_dim,n_windows,[],[])[0]
window_torch = helper.cut_density_windows_torch_padded(df, window_dim,n_windows)