import numpy as np 
import matplotlib.pyplot as plt
import torch
import os
import random
import conv_network as net
import neural_helper as helper
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)


parent_dir = os.path.dirname(os.getcwd())
# Path to the Density_profiles directory
density_profiles_dir = os.path.join(parent_dir, "Data_generation/Density_profiles")
tag = "parallel2025-05-16_18-08-55"
density_profiles_dir = os.path.join(density_profiles_dir, tag)

snippet = 7

names= [f for f in os.listdir(density_profiles_dir) if os.path.isfile(os.path.join(density_profiles_dir, f))]
for j in range(3,int(len(names)/snippet)):
    window_stack = []
    value_stack = []

    for i in names[j*snippet:(j+1)*snippet]:
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
        print("Stacked tensor shape:", window_tensor.shape)
    print(window_tensor.shape, value_tensor.shape)

    torch.save({"windows": window_tensor, "c1": value_tensor}, "training_data_test.pt")
    torch.cuda.empty_cache()





    path = os.getcwd()
    dataname = os.path.join(path, "training_data_test.pt")
    data = torch.load(dataname)
    inputs = data['windows']
    targets = data['c1']
    inputs = inputs.view(-1, 1, 213,213)  # Reshape to (N, C, H, W)
    inputs = inputs.float()
    targets = targets.view(-1,1,1 ,1)  # Reshape to (N, 1)
    dataset = torch.utils.data.TensorDataset(inputs, targets)

    data_size = len(dataset)
    train_size = int(0.7 * data_size)
    val_size = int(0.15 * data_size)
    test_size = data_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True,worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False,worker_init_fn=worker_init_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False,worker_init_fn=worker_init_fn)

    model = net.conv_neural_func7()
    model.load_state_dict(torch.load("2d_conv.pth"))
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 200

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    device = torch.device("cuda")
    model.to(device)



    train_loss = []
    validation_loss = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        ### Training loop
        for inputs,targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            ## update weights
            #print(loss.item())
            optimizer.step()
            running_loss += loss.item()
        running_loss /= len(train_loader)
        train_loss.append(running_loss)
        if epoch > 300 and epoch % 50 == 0:
            scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")

        ### Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        validation_loss.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")
    torch.cuda.empty_cache()



    torch.save(model.state_dict(), "2d_conv.pth")

    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("train_val_loss_conv"+str(j)+".png")
