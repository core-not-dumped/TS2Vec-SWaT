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
    "batch_size": 1,
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
    # score, _ = score_by_learnable_masking_random(model, proj_layer, pooling_layer, normal_train_dataloader, device, masking_len, progress=1.0)
    # warning_thr = np.percentile(score, 80)
    # crit_thr = np.percentile(score, 99)
    warning_thr = 12.16
    crit_thr = 15.92
    #for i in range(20):
    #    print(f'{i*5}th percentile: {np.percentile(score, i*5):.4f}')
    print(f"warning threshold: {warning_thr}")
    print(f"crit_threshold: {crit_thr}")
    '''
    0th percentile: 4.0078
    5th percentile: 7.6407
    10th percentile: 8.1838
    15th percentile: 8.5535
    20th percentile: 8.8576
    25th percentile: 9.1107
    30th percentile: 9.3428
    35th percentile: 9.5659
    40th percentile: 9.7766
    45th percentile: 9.9879
    50th percentile: 10.2027
    55th percentile: 10.4184
    60th percentile: 10.6353
    65th percentile: 10.8702
    70th percentile: 11.1392
    75th percentile: 11.4306
    80th percentile: 11.7630
    85th percentile: 12.1623
    90th percentile: 12.6992
    95th percentile: 13.6012
    '''

    total_anomaly_num = 0
    total_anomaly_detected_num = 0
    for x, y, ts in tqdm(normal_test_dataloader):
        x = x.to(device) # (B, T, C)
        B, T, C = x.shape

        # data augmentation
        ts_crop = ts[:,data_len-1:data_len*2-1] # (B, data_len)
        x_crop = x[:,data_len-1:data_len*2-1,:].clone() # (B, data_len, C)
        x_crop_changed, anomaly_mask, amp = inject_spike_anomaly(x_crop, 0.5, amp_start=5.0, amp_end=10.0, dur=10, change_sensor_num=change_sensor_num) # (B, data_len, C)
        x[:,data_len-1:data_len*2-1,:] = x_crop_changed

        # get anomaly score
        crop_out = proj_layer(x_crop_changed, no_mask=True) # (B, data_len, d_model)
        crop_out = last_repr_from_model(model, pooling_layer, crop_out) # (B, d_model)
        crop_changed_out = proj_layer(x_crop_changed.repeat_interleave(report_masking_len, dim=0), no_mask=False) # (B * report_masking_len, data_len, d_model)
        crop_changed_out = last_repr_from_model(model, pooling_layer, crop_changed_out) # (B * report_masking_len, d_model)
        crop_changed_out = crop_changed_out.view(B, report_masking_len, crop_changed_out.size(-1)) # (B, report_masking_len, d_model)
        anomaly_score = (crop_changed_out - crop_out.unsqueeze(1)).abs().sum(dim=-1) # (B, report_masking_len)
        anomaly_score = anomaly_score.mean(dim=-1) # (B,)

        # get timestamp wise anomaly score
        fold_x = x.unfold(1, data_len, 1)
        fold_x = fold_x.permute(0, 1, 3, 2) # (B, ~, data_len, C)
        fold_x = fold_x.contiguous().view(-1, data_len, channel_num) # (B * ~, data_len, C)
        fold_out = proj_layer(fold_x, no_mask=True) # (B * ~, data_len, d_model)
        fold_out = last_repr_from_model(model, pooling_layer, fold_out) # (B * ~, d_model)
        fold_out = fold_out.view(B, -1, fold_out.size(-1)) # (B, ~, d_model)

        masked_fold_x = fold_x.repeat_interleave(report_masking_len, dim=0) # (B * ~ * report_masking_len, data_len, C)
        masked_fold_out = proj_layer(masked_fold_x, no_mask=False) # (B * ~ * report_masking_len, data_len, d_model)
        masked_fold_out = last_repr_from_model(model, pooling_layer, masked_fold_out) # (B * ~ * report_masking_len, d_model)
        masked_fold_out = masked_fold_out.view(B, -1, report_masking_len, masked_fold_out.size(-1)) # (B, ~, report_masking_len, d_model)

        timestamp_anomaly_score = (masked_fold_out - fold_out.unsqueeze(2)).abs().sum(dim=-1).mean(dim=-1) # (B, ~)
        torch.set_printoptions(threshold=float('inf'))
        print(timestamp_anomaly_score)
        anomaly_sus_seq = timestamp_anomaly_score > warning_thr # (B, ~)
        # voting을 이용한 anomaly detection
        sus = (timestamp_anomaly_score > warning_thr).float().unsqueeze(1)                 # (B,1,W)
        kernel = torch.ones(1, 1, data_len, device=sus.device)           # (1,1,L)
        vote = F.conv1d(sus, kernel, padding=0).squeeze(1)               # (B,T)
        k = max(1, int(data_len * 0.95))                                  # 예: 80%
        anomaly_sus = (vote[:, :data_len] >= k)            # (B,data_len)

        anomaly_mask = anomaly_mask.any(dim=-1) # (B, data_len)
        anomaly_num = anomaly_mask.sum()
        anomaly_detected_num = ((anomaly_mask == anomaly_sus) & (anomaly_sus == 1)).sum()
        false_anomaly_detected_num = ((anomaly_mask != anomaly_sus) & (anomaly_sus == 1)).sum()
        total_anomaly_num += anomaly_num.item()
        total_anomaly_detected_num += anomaly_detected_num.item()

        print(
            f"anomaly num: {anomaly_num}, \n" + 
            f"detected anomaly num: {anomaly_detected_num}, \n" +
            f"false anomaly detection num: {false_anomaly_detected_num}, \n" +
            f"anomaly detection rate: {anomaly_detected_num / anomaly_num:.4f}"
        )

        real_report, real_anomaly_times = get_timewise_report(anomaly_mask, ts)
        model_report, model_anomaly_times = get_timewise_report(anomaly_sus, ts)
        for b, (r_real, t_real, r_model, t_model, a_score) in enumerate(zip(
            real_report, real_anomaly_times, model_report, model_anomaly_times, anomaly_score
        )):
            state = "CRITICAL" if a_score >= crit_thr else \
                    "WARNING" if a_score >= warning_thr else "NORMAL"
            print()
            print(f"=== Batch {b} ===")
            print(f"Anomaly_score: {a_score}, state: {state}")
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

