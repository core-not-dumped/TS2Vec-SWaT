from tqdm import tqdm

from model.inputProjection import *
from model.pooling import *
from model.customGPT import *
from model.customLSTM import *
from model.customDilatedCNN import *
from model.loss import *
from model.calc_score import *
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
normal_train_dataset = SWaTWindowDataset([f"./data/SWaT_processed/normal_{data_len * 3}_{i}.npz" for i in range(0, 8)])
normal_test_dataset = SWaTWindowDataset([f"./data/SWaT_processed/normal_{data_len * 3}_{i}.npz" for i in range(8, 10)])
normal_train_dataloader = torch.utils.data.DataLoader(normal_train_dataset, **train_data_loader_general_hyperparam)
normal_test_dataloader = torch.utils.data.DataLoader(normal_test_dataset, **test_data_loader_general_hyperparam)

channel_num = normal_train_dataset.x.shape[-1]
model = CustomDilatedCNN(d_model=d_model, n_layers=n_layers, kernel_size=3, dropout=dropout).to(device)
model.load_state_dict(torch.load(f"./model/{model_name}/199.pt", map_location=device, weights_only=True))
proj_layer = InputProjection_W_TimeSensorMasking(channel_num, d_model, time_masking_ratio=time_masking_ratio, sensor_masking_ratio=sensor_masking_ratio).to(device)
proj_layer.load_state_dict(torch.load(f"./model/{model_name}/199_proj.pt", map_location=device, weights_only=True))
pooling_layer = TS2VecMaxPooling(pooling_layer_num).to(device)

model.eval()
proj_layer.eval()
with torch.no_grad():
    #score, _ = score_by_learnable_masking_random(model, proj_layer, pooling_layer, normal_train_dataloader, device, masking_len, progress=1.0)
    #warning_thr = np.percentile(score, 80)
    #crit_thr = np.percentile(score, 99)
    warning_thr = 11.35
    crit_thr = 15.37
    #for i in range(20):
    #    print(f'{i*5}th percentile: {np.percentile(score, i*5):.4f}')
    print(f"warning threshold: {warning_thr}")
    print(f"crit_threshold: {crit_thr}")
    '''
    0th percentile: 4.6297
    5th percentile: 7.3645
    10th percentile: 7.8292
    15th percentile: 8.1726
    20th percentile: 8.4491
    25th percentile: 8.6964
    30th percentile: 8.9242
    35th percentile: 9.1391
    40th percentile: 9.3494
    45th percentile: 9.5640
    50th percentile: 9.7685
    55th percentile: 9.9841
    60th percentile: 10.2120
    65th percentile: 10.4514
    70th percentile: 10.7161
    75th percentile: 11.0179
    80th percentile: 11.3524
    85th percentile: 11.7629
    90th percentile: 12.3267
    95th percentile: 13.2665
    '''

    total_time_anomaly_num = 0
    total_time_anomaly_detected_num = 0
    for x, y, ts in tqdm(normal_test_dataloader):
        x = x.to(device) # (B, T, C)
        B, T, C = x.shape

        # data augmentation
        ts_crop = ts[:,data_len-1:data_len*2-1] # (B, data_len)
        x_crop = x[:,data_len-1:data_len*2-1,:].clone() # (B, data_len, C)
        x_crop_changed, anomaly_mask, amp = inject_spike_anomaly(x_crop, 0.5, amp_start=5.0, amp_end=10.0, dur=10, change_sensor_num=change_sensor_num) # (B, data_len, C)
        x[:,data_len-1:data_len*2-1,:] = x_crop_changed

        # get anomaly score ouptut -> (B,)
        anomaly_score = get_anomaly_score_one(x_crop_changed, proj_layer, model, pooling_layer, report_masking_len)
        
        # get timestamp wise anomaly score
        # idx, data_len - 1이 0가 오게 계산됨 output -> (B, T-data_len+1)
        timestamp_anomaly_score = get_timewise_anomaly_score_one(x, proj_layer, model, pooling_layer, report_masking_len, data_len)
        
        # voting을 이용한 anomaly detection
        sus = (timestamp_anomaly_score > warning_thr).float().unsqueeze(1)                 # (B,1,~)
        kernel = torch.ones(1, 1, data_len, device=sus.device)          # (1,1,L)
        vote = F.conv1d(sus, kernel, padding=0).squeeze(1)              # (B,T)
        k = max(1, int(data_len * 0.85))                                # 예: 80%
        time_anomaly_sus = (vote[:, :data_len] >= k)                    # (B,data_len)

        time_anomaly_num = anomaly_mask.any(dim=-1).sum()
        time_anomaly_detected_num = ((anomaly_mask.any(dim=-1) == time_anomaly_sus) & (time_anomaly_sus == 1)).sum()
        time_false_anomaly_detected_num = ((anomaly_mask.any(dim=-1) != time_anomaly_sus) & (time_anomaly_sus == 1)).sum()
        total_time_anomaly_num += time_anomaly_num.item()
        total_time_anomaly_detected_num += time_anomaly_detected_num.item()
        
        print(
            f"time anomaly num: {time_anomaly_num}, \n" + 
            f"time detected anomaly num: {time_anomaly_detected_num}, \n" +
            f"time false anomaly detection num: {time_false_anomaly_detected_num}, \n" +
            f"time anomaly detection rate: {time_anomaly_detected_num / time_anomaly_num:.4f} \n"
        )

        # get sensor wise anomaly score
        sensor_anomaly_score = get_sensorwise_anomaly_score_one(x_crop_changed, proj_layer, model, pooling_layer, time_anomaly_sus) # (B, C)
        sorted_score, sorted_idx = torch.sort(sensor_anomaly_score, dim=1, descending=True)
        sensor_anomaly_contribution = sorted_score / sorted_score.sum(dim=1, keepdim=True)

        real_timestamp_report, real_timestamp_anomaly_times = get_timewise_report(anomaly_mask.any(dim=-1), ts)
        model_timestamp_report, model_timestamp_anomaly_times = get_timewise_report(time_anomaly_sus, ts)
        model_sensor_report = get_sensorwise_report(sensor_anomaly_contribution, sorted_idx, ts, change_sensor_num)

        for b, (r_t_real, t_t_real, r_t_model, t_t_model, r_s_model, a_score) in enumerate(zip(
            real_timestamp_report, real_timestamp_anomaly_times,\
            model_timestamp_report, model_timestamp_anomaly_times,\
            model_sensor_report,\
            anomaly_score
        )):
            state = "CRITICAL" if a_score >= crit_thr else \
                    "WARNING" if a_score >= warning_thr else "NORMAL"
            print()
            print(f"=== Batch {b} ===")
            print(f'warning threshold: {warning_thr}, crit threshold: {crit_thr}')
            print(f"Anomaly_score: {a_score:.2f}, state: {state}")
            print("[Real]")
            print(r_t_real)
            print(f"Total anomaly time: {t_t_real}")
            print(f"Anomaly sensor: {anomaly_mask[b].any(dim=0).nonzero(as_tuple=False).squeeze().tolist()}")
            print()
            print("[Model]")
            print(r_t_model)
            print(f"Total anomaly time: {t_t_model}")
            if t_t_model:
                print("Sensor contribution score:")
                print(f"{r_s_model}\n")
            else:
                print("Sensor contribution score:")
                print(f"{r_s_model}\n")
            print("-" * 50)

    with open("./report.txt", "a") as f:
        write_and_print(f, \
            f"total time anomaly num: {total_time_anomaly_num},\n" + 
            f"total time detected anomaly num: {total_time_anomaly_detected_num},\n" + \
            f"time detection rate: {total_time_anomaly_detected_num / total_time_anomaly_num:.4f}\n")
        write_and_print(f, "-" * 50)

