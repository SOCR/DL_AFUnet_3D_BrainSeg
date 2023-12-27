import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from loss import DiceLoss
from skimage.segmentation import mark_boundaries
import nibabel as nib


diceloss = DiceLoss()
bceloss = nn.BCELoss()

def train(
    image_data,
    # val_data,
    model,
    num_epochs=100,
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


            # model = model.to(device)
            # X_batch = X_batch.to(device)
            # y_batch = y_batch.to(device)
            
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
        # val_loss_hist = (avg_epoch_val_loss / (batch_size))
        print(
            f"[epoch: {epoch_num+1}]",
            "[loss: ",
            f"{loss_hist:.4f}",
            "]"
        )
    plt.plot(loss_history)
    plt.plot(epoch_val_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss history")
    plt.show()

    torch.save(model, '/content/drive/My Drive/model_test/'+path)

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


def sample_test(path, test_set):
    model = torch.load(path, map_location='cuda')
    model.eval()

    a=0
    k=0
    for i in range(0, 500):
        # print(i)
        k+=1
        X = test_set['X_train'][i]
        gnd = test_set['y_train'][i]
        gnd = gnd.view(1, 1, 256, 256)
        gnd = gnd.to(device='cuda')
        gnd = (gnd > 0.5)

        X = X.view(1, 3, 256, 256)
        X = X.to(device='cuda')

        out = model(X)
        out = (out > 0.5)
        a += diceloss(out, gnd) # total dice loss 

        out = np.array((out.view(256, 256)).cpu().detach().numpy())

    return (a/k)


def sample_test_3d(path, test_set):
    model = torch.load(path)
    model.eval()

    a=0
    k=0
    for i in range(0, 50):
        k+=1
        X = test_set['X_train'][i]
        gnd = test_set['y_train'][i]
        gnd = gnd.view(1, 1, 128, 128, 64)
        gnd = gnd.to(device='cuda')

        X = X.view(1, 4, 128, 128, 64)
        X = X.to(device='cuda')

        out = model(X)
        out = (out > 0.5)
        a += diceloss(out, gnd)

    return (a/k)



def boundary_compare(path1, path2, original, gnd):

    model1 = torch.load(path1, map_location=torch.device('cpu'))
    model1.eval()
    model2 = torch.load(path2, map_location=torch.device('cpu'))
    model2.eval()

    gnd = gnd.view(1, 1, 256, 256)
    gnd = np.array((gnd.view(256, 256)).cpu().detach().numpy())
    gnd = (gnd>0.5)
    original = original.view(1, 3, 256, 256)

    out1 = model1(original)
    out1 = (out1 > 0.5)
    out1 = np.array((out1.view(256, 256)).cpu().detach().numpy())

    out2 = model2(original)
    out2 = (out2 > 0.5)
    out2 = np.array((out2.view(256, 256)).cpu().detach().numpy())

    original = original.view(256, 256, 3)
    original = ((original.permute(1, 2, 0))).cpu().detach().numpy() / 255
    edges_pz = show_boundaries(original, gnd, out1, out2)

    return edges_pz


def plot_ci(x):

    x = [0,1,2,3]
    # y = [99.55, 97.57, 95.27, 96.85, 93.98]
    list_y = [[0.7304, 0.7413, 0.7381, 0.7193, 0.6883],
              [0.8166, 0.8187, 0.8140, 0.8163, 0.7954],
              [0.8728, 0.8613, 0.8547, 0.8422, 0.8307],
              [0.8619, 0.8586, 0.8392, 0.8338, 0.8369]
    ]
    alpha_values = [0, 1, 2, 3]
    list_nll = []
    for i in range(4):
      list_nll.append(np.mean(list_y[i]))
    list_low = []
    list_up = []
    for i in range(4):
      ci = 1.96 * np.std(list_y[i])/np.sqrt(5)
      list_low.append(np.mean(list_y[i])-ci)
      list_up.append(np.mean(list_y[i])+ci)
      print(np.mean(list_y[i]))
      print(ci)
    fig, ax = plt.subplots()
    ax.plot(x, list_nll)
    ax.fill_between(x, list_low, list_up, color='b', alpha=.1)

    # plt.plot(x, list_nll, label = "NLL vs alpha values")
    plt.xlabel("The iteration depth")
    plt.ylabel("Dice Score")
    plt.xticks(x, alpha_values)
    plt.legend()
    plt.show()


def output_save_3d(model, original, gnd, save_path):
    
    original = original.view(1, 4, 128, 128, 64)
    original = original.to(device='cuda')

    out = model(original)

    original = (original[0, 1, :, :, :].view(128, 128 ,64)).cpu().detach().numpy()
    out = (out.view(128, 128, 64)).cpu().detach().numpy()
    gnd = (gnd.view(128, 128, 64)).cpu().detach().numpy()
    original = np.array(original, dtype=np.float32)
    out = (out>0.5)
    out = np.array(out, dtype=np.float32)
    gnd = np.array(gnd, dtype=np.float32)

    gnd *= 2
    out += gnd

    new_image1 = nib.Nifti1Image(original, affine=np.eye(4))
    new_image2 = nib.Nifti1Image(out, affine=np.eye(4))
    new_image3 = nib.Nifti1Image(gnd, affine=np.eye(4))
    new_image1.header.get_xyzt_units()
    new_image1.to_filename('/content/drive/My Drive/models/t1.nii')
    new_image2.header.get_xyzt_units()
    new_image2.to_filename('/content/drive/My Drive/models/mask.nii')
    new_image3.header.get_xyzt_units()
    new_image3.to_filename('/content/drive/My Drive/models/gnd.nii')

    return new_image1, new_image2, new_image3
