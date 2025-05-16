import torch
import numpy as np 
import pandas as pd
import os
import neural_helper as helper

df = pd.DataFrame({
    'x': np.zeros(10*10),
    'y': np.zeros(10*10),
    'rho': np.zeros(10*10),
    'muloc': np.zeros(10*10),
})
xa = np.linspace(1, 10, 10)
ya = np.linspace(1, 10, 10)
z = 0
for i,x in enumerate(xa):
    for j,y in enumerate(ya):

        df['x'][i*10+j] =x
        df['y'][i*10+j] =y
        df['muloc'][i*10+j] =z
        df['rho'][i*10+j] =z
        z+=1
print(len(df))
window,center =helper.build_training_data_torch(df, 5, 10)
print(window)
print(center)
print(window.shape,center.shape)

