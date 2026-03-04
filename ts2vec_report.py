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
normal_train_dataset = SWaTWindowDataset([f"./data/SWaT_processed/normal_{i}.npz" for i in range(1, 8)])
normal_test_dataset = SWaTWindowDataset([f"./data/SWaT_processed/normal_{i}.npz" for i in range(8, 10)])
normal_train_dataloader = torch.utils.data.DataLoader(normal_train_dataset, **data_loader_general_hyperparam)
normal_test_dataloader = torch.utils.data.DataLoader(normal_test_dataset, **data_loader_general_hyperparam)

model = torch.load(f'./model/{model_name}/10.pt')
proj_layer = torch.load(f'./model/{model_name}/10_proj.pt')
pooling_layer = TS2VecMaxPooling(pooling_layer_num).to(device)

model.eval()
proj_layer.eval()
with torch.no_grad():
    score, _ = score_by_learnable_masking_sequential(model, proj_layer, pooling_layer, normal_train_dataloader, device, masking_len)
    thr = np.percentile(score, 99)

    for x, y, ts in tqdm(normal_test_dataloader):
        x = x.to(device) # (B, T, C)

        # data augmentation
        x = augment_view_return1(x, data_len) # (B, data_len, C)
        x, anomaly_mask, amp = inject_spike_anomaly(x, anomaly_ratio, amp_start=1.0, amp_end=2.0, dur=15, seed=None) # (B, data_len, C)

        out = proj_layer(x, no_mask=True) # (B, data_len, d_model)
        out = last_repr_from_model(model, pooling_layer, out) # (B, d_model)

        masked_x = x.unsqueeze(1).repeat(1, masking_len, 1, 1) # (B, masking_len, data_len, C)
        mask = torch.ones(data_len, data_len, device=x.device).bool()
        mask.fill_diagonal_(0)
        masked_x = (masked_x * mask[None, :, :, None]).reshape(-1, data_len, out.size(-1))
        masked_out = proj_layer(masked_x, no_mask=True)
        masked_out = last_repr_from_model(model, pooling_layer, masked_out) # (B * data_len, d_model)
        masked_out = masked_out.reshape(x.size(0), data_len, masked_out.size(-1)) # (B, data_len, d_model)

        anomaly_score = (masked_out - out.unsqueeze(1)).abs().sum(dim=-1) # (B, data_len(masking_len))
        anomaly_score > thr

