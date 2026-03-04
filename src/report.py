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