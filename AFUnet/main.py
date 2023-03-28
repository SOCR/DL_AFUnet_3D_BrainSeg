import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from loss import DiceLoss
from skimage.segmentation import mark_boundaries


def train(
    image_data,
    # val_data,
    model,
    num_epochs=200,
    batch_size=20,
    learning_rate=5e-4,
    criterion = DiceLoss(),
    path = 'AFUnet.pth',
    device: torch.device = torch.device("cuda"),
):
    X_train = image_data['X_train']
    y_train = image_data['y_train']
    # X_val = val_data['X_train']
    # y_val = val_data['y_train']
    # X_val = X_val.to(device)
    # y_val = y_val.to(device)

    print("Training started...")

    model = model.to(device)
    loss_history = []
    val_loss_history = []
    optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, betas=(0.9, 0.995), eps=1e-9
        )
    iteration = 0
    num_train = X_train.shape[0]
    for epoch_num in range(num_epochs):
        epoch_loss = []
        epoch_val_loss = []
        model.train()
        idx = torch.randperm(num_train)
        for i in range(num_train//batch_size):
            
            batch_mask = idx[i*batch_size: i*batch_size+batch_size]
            X_batch = X_train[batch_mask].to(device)
            y_batch = y_train[batch_mask].to(device)


            model = model.to(device)
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()

            gnd = y_batch
            pred = model(X_batch)
            loss = criterion(pred, gnd)
            # print(loss)
            
            epoch_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            iteration = iteration + 1
        
        # pred_val = model(X_val)
        # loss_val = criterion(pred_val, y_val)
        # epoch_val_loss.append(loss_val.item())

        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        loss_hist = (avg_epoch_loss / (batch_size))
        loss_history.append(loss_hist)
        # avg_epoch_val_loss = sum(epoch_val_loss) / len(epoch_val_loss)
        # val_loss_hist = avg_epoch_val_loss / (batch_size)
        print(
            f"[epoch: {epoch_num+1}]",
            "[loss: ",
            f"{loss_hist:.4f}",
            "]"
        )
    plt.plot(loss_history)
    # plt.plot(epoch_val_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss history")
    plt.show()

    torch.save(model, '/content/drive/My Drive/models/'+path) # Change the path to your own

    return model


def show_boundaries(original, gnd, out_1, out_2):
    
    edges_pz1 = mark_boundaries(original, gnd, color=(1,0,0), mode='outer') # Red
    edges_pz2 = mark_boundaries(original, out_1, color=(0,1,0), mode='outer') # Green
    edges_pz3 = mark_boundaries(original, out_2, color=(0,0,1), mode='outer') # Blue

    edges_pz = np.ones((256, 256, 3))
    edges_pz[:, :, 0] = edges_pz1[:, :, 0]
    edges_pz[:, :, 1] = edges_pz2[:, :, 1]
    edges_pz[:, :, 1][edges_pz1[:, :, 0] == 1] = 0
    edges_pz[:, :, 2] = edges_pz3[:, :, 2]
    edges_pz[:, :, 2][edges_pz1[:, :, 0] == 1] = 0
    edges_pz[:, :, 2][edges_pz2[:, :, 1] == 1] = 0
    
    return edges_pz

def sample_test():
    pass

