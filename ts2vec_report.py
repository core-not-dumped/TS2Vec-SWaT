from datetime import datetime
import os

from tqdm import tqdm

from model.inputProjection import *
from model.pooling import *
from model.customGPT import *
from model.customLSTM import *
from model.customDilatedCNN import *
from model.loss import *
from data.dataset import *
from data.augmentation import *
from src.hyperparam import *

def write_and_print(f, msg):
    print(msg)
    f.write(msg + "\n")
    f.flush()

data_loader_general_hyperparam = {
    "batch_size": batch_size,
    "shuffle": True,
    "num_workers": cpu_num,
}
normal_test_dataset = SWaTWindowDataset([f"./data/SWaT_processed/normal_{i}.npz" for i in range(8, 10)])
normal_test_dataloader = torch.utils.data.DataLoader(normal_test_dataset, **data_loader_general_hyperparam)

model = torch.load(f'./model/{model_name}/10.pt')
proj_layer = torch.load(f'./model/{model_name}/10_proj.pt')

model.eval()
proj_layer.eval()
for x, y, ts in tqdm(normal_test_dataloader):
    x = x.to(device) # (B, T, C)

    # data augmentation
    x = augment_view_return1(x, data_len) # (B, data_len, C)
    x = augment_anomaly(x, anomaly_ratio) # (B, data_len, C)



    # pooling + timestamp masking
    x1 = proj_layer(x1) # (B, data_len, d_model)
    x2 = proj_layer(x2) # (B, data_len, d_model)

    # Dilated Convolution (Transformer, LSTM, CNN..., main model)
    out1 = model(x1) # (B, data_len, d_model)
    out2 = model(x2) # (B, data_len, d_model)

    # pooling layer
    outs1 = pooling_layer(out1)
    outs2 = pooling_layer(out2)

    # loss backward
    loss = criterion(outs1, outs2)
    optimizers.zero_grad()
    loss.backward()
    optimizers.step()

    losses.append(loss.item())