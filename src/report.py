import datetime
import torch

def get_timewise_report(mask, ts):
    '''
    mask: (B, T)
    ts: (B, T)
    '''
    B, T = mask.shape
    report = []
    anomaly_times = []

    mask = torch.cat([mask, torch.zeros(B,1, device=mask.device)], dim=1)

    for b in range(B):
        sample_report = []
        in_anomaly = False
        start_t = None
        anomaly_time = 0

        for t in range(T+1):
            if mask[b,t] == 1:  anomaly_time += 1

            if mask[b,t] == 1 and not in_anomaly:
                in_anomaly = True
                start_t = ts[b,t]

            elif mask[b,t] == 0 and in_anomaly:
                end_t = ts[b,t-1]
                sample_report.append(
                    f"{datetime.datetime.fromtimestamp(start_t.item() / 1e9)} ~ {datetime.datetime.fromtimestamp(end_t.item() / 1e9)}: anomaly detected"
                )
                in_anomaly = False

        report.append("\n".join(sample_report))
        anomaly_times.append(anomaly_time)

    return report, anomaly_times

def get_sensorwise_report(contribution, sorted_idx, ts, change_sensor_num, sensors_name, mean, std, x_crop_changed, time_anomaly_mask):
    '''
    contribution: (B, C)
    sorted_idx: (B, C)
    ts: (B, T)
    sensors_name: (C,)
    mean, std: (1, C)
    time_anomaly_mask: (B, data_len) -> if true, then the timestamp is detected as anomaly, else normal
    '''
    B, C = contribution.shape
    report = []
    for b in range(B):
        sample_report = []
        for c in range(change_sensor_num):
            sensor_report = []
            sensor_idx = sorted_idx[b,c].item()
            contrib_score = contribution[b,c].item()
            if contrib_score < 0.05: break
            sensor_report.append(f"Sensor {sensors_name[sensor_idx]}: contribution score {contrib_score:.4f}")

            mask = time_anomaly_mask[b] # (data_len,)
            if mask.any():
                x_crop = x_crop_changed[b][mask, sensor_idx]
                mean_val = mean[0, sensor_idx].item()
                std_val = std[0, sensor_idx].item()
                x_crop_denorm = x_crop * std_val + mean_val
                sensor_report.append(f"values ranged from {x_crop_denorm.min().item():.2f} to {x_crop_denorm.max().item():.2f}")
            sample_report.append(", ".join(sensor_report))
        report.append("\n".join(sample_report))
    return report