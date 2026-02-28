import numpy as np
import pandas as pd
from pathlib import Path
from src.hyperparam import *

LABEL_COL = "Normal/Attack"


class StandardScaler:
    def fit(self, x):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True) + 1e-8
        return self
    def transform(self, x):
        return (x - self.mean) / self.std


def sliding_window(x, y, ts, window, stride):
    xs, ys, tss = [], [], []
    one_sec = 1_000_000_000
    for s in range(0, len(x) - window + 1, stride):
        if ts[s + window - 1] - ts[s] != (window - 1) * one_sec:  # timestamp 간격이 일정하지 않으면 skip
            continue
        xs.append(x[s:s + window])
        ys.append(int(y[s:s + window].max() > 0))
        tss.append(ts[s:s + window])
    return (
        np.stack(xs),
        np.array(ys, dtype=np.int64),
        np.stack(tss),
    )


if __name__ == "__main__":
    data_sp = "attack"  # "normal" or "attack"
    data_cut = normal_data_cut if data_sp == "normal" else attack_data_cut

    data_dir = Path("./data/SWaT")
    df = pd.read_csv(data_dir / f"{data_sp}.csv")
    out_dir = Path("./data/SWaT_processed")
    out_dir.mkdir(exist_ok=True)
    drop_cols = ['MV101','AIT201','MV201','P201','P202','P204','MV303']

    # timestamp, label, features 분리
    ts = pd.to_datetime(df[" Timestamp"].str.strip(), format="%d/%m/%Y %I:%M:%S %p").astype(np.int64).values
    y = (df[LABEL_COL].values != "Normal").astype(np.int64) # y: "Normal" / "Attack" -> 0 / 1
    x = df.drop(columns=[LABEL_COL, ' Timestamp'] + drop_cols, errors="ignore").values.astype(np.float32)
    

    # 정규화
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)

    # sliding window 적용
    x_win, y_win, ts_win = sliding_window(x, y, ts, data_window_size, data_stride)

    # 데이터 분할
    N = len(x_win)
    for i in range(data_cut):
        split_s = int(N // data_cut * i)
        split_e = int(N // data_cut * (i + 1))
        x = x_win[split_s:split_e]
        y = y_win[split_s:split_e]
        ts = ts_win[split_s:split_e]

        np.savez_compressed(
            out_dir / f"{data_sp}_{i}.npz",
            x=x, y=y, ts=ts
        )

        print("x:", x.shape, "y:", y.shape, "ts:", ts.shape)

    print("DONE")