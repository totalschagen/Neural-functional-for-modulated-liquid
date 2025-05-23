import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
import pandas as pd
import os
import math
parent_dir = os.path.dirname(os.getcwd())
# Path to the Density_profiles directory
density_profiles_dir = os.path.join(parent_dir, "Density_profiles")

# Count the number of files in the directory

def load_df(name,tag):
    name = os.path.join(parent_dir, "Density_profiles",tag, name)
    print("Reading file: ", name)
    df = pd.read_csv(name,delimiter = " ")

    extra=[]
    try:
        nperiod = (list(df.columns)[4])
        df.drop(columns=[nperiod],inplace=True)
        nperiod = int(nperiod)
        extra.append(nperiod)
    except:
        print("No nperiod")
    try:
        mu = (list(df.columns)[4])

        df.drop(columns=[mu],inplace=True)
        mu = float(mu)
        extra.append(mu)
    except:
        print("No mu")
    try:
        packing = (list(df.columns)[4])
        df.drop(columns=[packing],inplace=True)
        packing = float(packing)
        extra.append(packing)
    except:
        print("No packing")
    return df, extra

tag = "parallel2025-05-09_18-33-17"
num = 4
density_profiles_dir = os.path.join(parent_dir, "Density_profiles",tag)
names= [f for f in os.listdir(density_profiles_dir) if os.path.isfile(os.path.join(density_profiles_dir, f))]
slice = names[50:50+num]
rows = len(slice)//2
cols = 2

## create subplots
fig, axs = plt.subplots(rows, cols, figsize=(11,10*rows/2))
fig.suptitle("2D Density profile")
for i, name in enumerate(slice):
    # Get the current dataframe
    df, extra = load_df(name, tag)
    print(extra) 
    # Calculate the row and column index for the subplot
    period, mu  = extra[0], extra[1]
    row = i // cols
    col = i % cols

    shape = np.sqrt(len(df)).astype(int)
    xi = df["x"].values.reshape(shape, shape)
    yi = df["y"].values.reshape(shape,shape)
    rho = df["rho"].values.reshape(shape,shape)
    
    pcm = axs[row, col].pcolormesh(xi, yi, rho, shading='auto', cmap='viridis',label=f"Nperiod = {period}, mu = {mu}")
    # axs[i, 0].set_title(f"Nperiod = {period}, mu = {mu}")
    axs[row, col].set_xlabel(r"$x/\sigma$")
    axs[row, col].set_ylabel(r"$y/\sigma$")
    label = f"Nperiod = {period}, mu = {mu}"
    patch = mpatches.Patch(color=pcm.cmap(0.5), label=label)
    axs[row, col].legend(handles=[patch])

#     x_values = df.groupby("x").mean().reset_index()["x"]
#     rho_values = df.groupby("x").mean().reset_index()["rho"]
    
#     axs[i, 1].plot(x_values, rho_values, label="rho")
# #    axs[i, 1].plot(x_values, potential(x_values, period, 15, 3), label="potential")
#     axs[i, 1].set_title(name)
#     axs[i, 1].set_xlabel(r"$x/\sigma$")
#     axs[i, 1].set_ylabel(r"$\rho$")

    fig.colorbar(pcm, ax=axs[row,col], label=r"$\rho^{(1)}(\mathbf{x})$")
# ➡️ 4. One shared colorbar for all density plots

plt.tight_layout()
plt.savefig("plot.png")