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
from src.report import *

def write_and_print(f, msg):
    print(msg)
    f.write(msg + "\n")
    f.flush()

train_data_loader_general_hyperparam = {
    "batch_size": batch_size,
    "shuffle": True,
    "num_workers": cpu_num,
}
test_data_loader_general_hyperparam = {
    "batch_size": batch_size,
    "shuffle": True,
    "num_workers": cpu_num,
}
normal_train_dataset = SWaTWindowDataset([f"./data/SWaT_processed/normal_{data_len * 3}_{i}.npz" for i in range(1, 8)])
normal_test_dataset = SWaTWindowDataset([f"./data/SWaT_processed/normal_{data_len * 3}_{i}.npz" for i in range(8, 10)])
normal_train_dataloader = torch.utils.data.DataLoader(normal_train_dataset, **train_data_loader_general_hyperparam)
normal_test_dataloader = torch.utils.data.DataLoader(normal_test_dataset, **test_data_loader_general_hyperparam)

channel_num = normal_train_dataset.x.shape[-1]
model = CustomDilatedCNN(d_model=d_model, n_layers=n_layers, kernel_size=3, dropout=dropout).to(device)
model.load_state_dict(torch.load(f"./model/{model_name}/30.pt", map_location=device, weights_only=True))
proj_layer = InputProjection_W_TimeSensorMasking(channel_num, d_model, time_masking_ratio=time_masking_ratio, sensor_masking_ratio=sensor_masking_ratio).to(device)
proj_layer.load_state_dict(torch.load(f"./model/{model_name}/30_proj.pt", map_location=device, weights_only=True))
pooling_layer = TS2VecMaxPooling(pooling_layer_num).to(device)

model.eval()
proj_layer.eval()
with torch.no_grad():
    score, _ = score_by_learnable_masking_random(model, proj_layer, pooling_layer, normal_train_dataloader, device, masking_len, progress=1.0)
    thr = np.percentile(score, 99)
    print(f"threshold: {thr}")

    total_anomaly_num = 0
    total_anomaly_detected_num = 0
    for x, y, ts in tqdm(normal_test_dataloader):
        x = x.to(device) # (B, T, C)
        B, T, C = x.shape

        # data augmentation
        ts_crop = ts[:,data_len-1:data_len*2-1] # (B, data_len)
        x_crop = x[:,data_len-1:data_len*2-1,:] # (B, data_len, C)
        x_crop, anomaly_mask, amp = inject_spike_anomaly(x_crop, anomaly_ratio, amp_start=1.0, amp_end=2.0, dur=15) # (B, data_len, C)
        x[:,data_len-1:data_len*2-1,:] = x_crop

        fold_x = x.unfold(1, data_len, 1) # (B, ~, data_len, C)
        fold_x = fold_x.permute(0, 1, 3, 2)
        fold_x = fold_x.contiguous().view(-1, data_len, channel_num) # (B * ~(65), data_len, C)
        out = proj_layer(fold_x, no_mask=True) # (B * ~, data_len, d_model)
        outs = last_repr_from_model(model, pooling_layer, out) # (B * ~, d_model)
        outs = outs.view(B, -1, outs.size(-1)) # (B, ~, d_model)

        masked_fold_x = fold_x.repeat_interleave(report_masking_len, dim=0) # (B * ~ * report_masking_len, data_len, C)
        masked_out = proj_layer(masked_fold_x, no_mask=False) # (B * ~ * report_masking_len, data_len, d_model)
        masked_outs = last_repr_from_model(model, pooling_layer, masked_out) # (B * ~ * report_masking_len, d_model)
        masked_outs = masked_outs.view(B, -1, report_masking_len, masked_outs.size(-1)) # (B, ~, report_masking_len, d_model)

        anomaly_score = (masked_outs - outs.unsqueeze(2)).abs().sum(dim=-1).mean(dim=-1) # (B, ~)
        print(anomaly_score)
        anomaly_sus_seq = anomaly_score > thr # (B, ~)
        anomaly_sus = anomaly_sus_seq[:, :data_len] & anomaly_sus_seq[:, data_len-1:2*data_len-1] # (B, data_len)

        anomaly_num = anomaly_mask.sum()
        anomaly_detected_num = ((anomaly_mask.sum(-1) == anomaly_sus) & (anomaly_sus == 1)).sum()
        false_anomaly_detected_num = ((anomaly_mask.sum(-1) != anomaly_sus) & (anomaly_sus == 1)).sum()
        total_anomaly_num += anomaly_num.item()
        total_anomaly_detected_num += anomaly_detected_num.item()
        print(
            f"anomaly num: {anomaly_num}, \n" + 
            f"detected anomaly num: {anomaly_detected_num}, \n" +
            f"false anomaly detection num: {false_anomaly_detected_num}, \n" +
            f"anomaly detection rate: {anomaly_detected_num / anomaly_num:.4f}"
        )

        real_report, real_anomaly_times = get_timewise_report(anomaly_mask.sum(dim=-1), ts)
        model_report, model_anomaly_times = get_timewise_report(anomaly_sus, ts)
        for b, (r_real, t_real, r_model, t_model) in enumerate(zip(real_report, real_anomaly_times, model_report, model_anomaly_times)):
            print(f"=== Batch {b} ===")
            print("[Real]")
            print(r_real)
            print(f"Total anomaly time: {t_real}")
            print()
            print("[Model]")
            print(r_model)
            print(f"Total anomaly time: {t_model}")
            print("-" * 50)

    with open("./report.txt", "a") as f:
        write_and_print(f, \
            f"total anomaly num: {total_anomaly_num},\n" + 
            f"total detected anomaly num: {total_anomaly_detected_num},\n" + \
            f"detection rate: {total_anomaly_detected_num / total_anomaly_num:.4f}\n")
        write_and_print(f, "-" * 50)

