import numpy as np
import torch
import os
import conv_network as net
import matplotlib.pyplot as plt
import neural_helper as helper
import pandas

model = net.conv_neural_func7()
model.load_state_dict(torch.load("2d_conv.pth"))
model.to(torch.device("cuda"))

parent_dir = os.path.dirname(os.getcwd())
# Path to the Density_profiles directory
density_profiles_dir =  "/home/hagen/Documents/master_project/Project/Data_generation/Density_profiles"
tag = "current_train"
density_profiles_dir = os.path.join(density_profiles_dir, tag)





_,_,window_dims,_ =next(model.children()).weight.shape


names= [f for f in os.listdir(density_profiles_dir) if os.path.isfile(os.path.join(density_profiles_dir, f))]
for name in names:
    df,_ = helper.load_df(name,density_profiles_dir)
    n_windows=int(np.sqrt(len(df))/window_dims)
    window,value = helper.cut_density_windows_torch_padded(df, window_dims)
    model.eval()
    outputs= []
    with torch.no_grad():
        for input in window:
            input = input.unsqueeze(0)
            print(input.shape)
            input = input.to(torch.device("cuda"))
            output = model(input)
            output = output.cpu().numpy()
            outputs.append(output)

    print(np.array(outputs).shape)
    c1_reconstruct = helper.reconstruct_values(torch.tensor(np.array(outputs)), 18,window_dims)
    cutout_reconstruct = helper.reconstruct_values(value, 18,window_dims)
    print(c1_reconstruct.shape)
    print(value.shape)

    df["muloc"]=np.log(df["rho"])+df["muloc"]
    mulocmatrix = df.pivot(index='y', columns='x', values='muloc').values
    fig, ax = plt.subplots(2,2, figsize=(18, 6))
    fig.suptitle(name)
    a = ax[1,0].imshow(c1_reconstruct)
    fig.colorbar(a, ax=ax[1,0])
    ax[1,0].set_title("Reconstructed c1")
    b = ax[0,0].imshow(mulocmatrix)
    fig.colorbar(b, ax=ax[0,0])
    ax[0,0].set_title("Original c1")
    c = ax[0,1].imshow(cutout_reconstruct)
    fig.colorbar(c, ax=ax[0,1])
    ax[0,1].set_title("Cutout c1")
    d = ax[1,1].imshow(c1_reconstruct-cutout_reconstruct)
    fig.colorbar(d, ax=ax[1,1])
    ax[1,1].set_title("difference")
    save_path = os.path.join(os.getcwd(), "inference_out",name+".png")
    plt.savefig(save_path)
    plt.close()
