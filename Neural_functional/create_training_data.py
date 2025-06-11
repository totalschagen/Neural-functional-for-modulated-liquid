import torch
import numpy as np 
import pandas as pd
import os
import neural_helper as helper
parent_dir = os.path.dirname(os.getcwd())
# Path to the Density_profiles directory
density_profiles_dir = os.path.join(parent_dir, "Data_generation/Density_profiles")
tag = "parallel2025-05-16_18-08-55"
density_profiles_dir = os.path.join(density_profiles_dir, tag)


names= [f for f in os.listdir(density_profiles_dir) if os.path.isfile(os.path.join(density_profiles_dir, f))]

window_stack = []
value_stack = []

for i in names[:]:
    df,nperiod = helper.load_df(i,density_profiles_dir)
    # Get the number of rows and columns in the DataFrame
    df["muloc"]=np.log(df["rho"])+df["muloc"]
    if np.sum(df["rho"])< 0.1:
        continue
    window,value= helper.build_training_data_torch(df, 2.5, 15)
    window_stack.append(window)
    value_stack.append(value)
window_tensor = window_stack[0]
value_tensor = value_stack[0]
for i in range(1,len(window_stack)):
    window_tensor = torch.cat((window_tensor, window_stack[i]), 0)
    value_tensor = torch.cat((value_tensor, value_stack[i]), 0)
print(window_tensor.shape, value_tensor.shape)

torch.save({"windows": window_tensor, "c1": value_tensor}, "training_data_test.pt")


