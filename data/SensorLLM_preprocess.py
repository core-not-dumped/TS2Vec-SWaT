import numpy as np
from dataclasses import dataclass
from typing import List, Literal, Dict, Tuple, Optional
import pandas as pd
import json
from tqdm import tqdm

from src.hyperparam import *

Trend = Literal["stable", "increasing", "decreasing"]

@dataclass
class Seg:
    start_s: float
    end_s: float
    trend: Trend

def _moving_avg(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    win = int(win)
    k = np.ones(win, dtype=np.float32) / win
    return np.convolve(x, k, mode="same")

def _segment_trends_with_tolerance(
    y: np.ndarray,
    t: np.ndarray,
    stable_eps: float,
    slope_tau: float,
    min_seg_s: float,
) -> List[Seg]:
    """
    Segment a 1D series y[t] into stable/increasing/decreasing segments.
    stable_eps: if |Δy| <= stable_eps => stable step
    slope_tau:  if Δy > slope_tau => increasing, Δy < -slope_tau => decreasing
               (note: stable_eps should usually be <= slope_tau)
    min_seg_s: merge segments shorter than this duration
    """
    y = np.asarray(y, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)
    if len(y) <= 1:
        return [Seg(float(t[0]), float(t[-1]), "stable")]

    dy = np.diff(y)

    # step-level label
    step = np.full(len(dy), "stable", dtype=object)
    step[dy > slope_tau] = "increasing"
    step[dy < -slope_tau] = "decreasing"
    # tighten stable region: if abs(dy) <= stable_eps -> stable
    step[np.abs(dy) <= stable_eps] = "stable"

    segs: List[Seg] = []
    start_i = 0
    for i in range(1, len(step)):
        if step[i] != step[start_i]:
            segs.append(Seg(float(t[start_i]), float(t[i]), step[start_i]))  # end at t[i]
            start_i = i
    segs.append(Seg(float(t[start_i]), float(t[-1]), step[start_i]))

    # merge too-short segments into previous (simple + stable)
    merged: List[Seg] = []
    for s in segs:
        dur = s.end_s - s.start_s
        if not merged:
            merged.append(s)
        elif dur < min_seg_s:
            prev = merged[-1]
            merged[-1] = Seg(prev.start_s, s.end_s, prev.trend)
        else:
            merged.append(s)

    # merge consecutive same trends
    compact: List[Seg] = []
    for s in merged:
        if not compact or compact[-1].trend != s.trend:
            compact.append(s)
        else:
            prev = compact[-1]
            compact[-1] = Seg(prev.start_s, s.end_s, prev.trend)

    return compact

def _round_time(x: float, quantum: float) -> float:
    return float(np.round(x / quantum) * quantum)

def build_sensorllm_style_answer_float(
    *,
    # Provide either wav or already-reduced 1D series
    series: np.ndarray = None,
    series_times: np.ndarray = None,
    # segmentation controls
    smooth_win: int = 2,
    time_quantum_s: float = 0.02,     # example aligns with 0.02s steps (adjust)
    stable_eps_quantile: float = 0.40, # stable tolerance from dy distribution
    slope_tau_quantile: float = 0.70,  # inc/dec threshold from dy distribution
    min_seg_s: float = 0.04,           # merge segments shorter than this
    # meta for the summary paragraph
    sensor_desc: str = "normalized fan microphone",
    mode: Literal["trend+summary", "trend_only", "summary_only"] = "trend+summary",
) -> str:
    """
    Produce SensorLLM-like Stage1 answer:
    - list of segments with 'stable/increasing/decreasing'
    - counts
    - narrative summary with totals + dominant trend
    Works for wav (via RMS envelope) or any 1D series.
    """
    tt = np.asarray(series_times, dtype=np.float32)
    y = _moving_avg(np.asarray(series, dtype=np.float32), smooth_win)

    # thresholds from dy distribution (robust + data-dependent)
    dy = np.diff(y) if len(y) > 1 else np.array([0.0], dtype=np.float32)
    abs_dy = np.abs(dy) + 1e-12
    stable_eps = float(np.quantile(abs_dy, stable_eps_quantile))
    slope_tau = float(np.quantile(abs_dy, slope_tau_quantile))

    # ensure stable_eps <= slope_tau (usually desirable)
    stable_eps = min(stable_eps, slope_tau)

    segs = _segment_trends_with_tolerance(
        y=y, t=tt, stable_eps=stable_eps, slope_tau=slope_tau, min_seg_s=min_seg_s
    )

    # round times
    rounded: List[Seg] = []
    for s in segs:
        st = _round_time(s.start_s, time_quantum_s)
        en = _round_time(s.end_s, time_quantum_s)
        if en <= st:
            continue
        rounded.append(Seg(st, en, s.trend))
    if not rounded:
        # fallback
        rounded = [Seg(_round_time(float(tt[0]), time_quantum_s),
                       _round_time(float(tt[-1]), time_quantum_s), "stable")]
    # 마지막 초 보정
    rounded[-1] = Seg(rounded[-1].start_s, rounded[-1].end_s + time_quantum_s, rounded[-1].trend)
 
    # counts and total durations
    counts: Dict[Trend, int] = {"stable": 0, "decreasing": 0, "increasing": 0}
    total_dur: Dict[Trend, float] = {"stable": 0.0, "decreasing": 0.0, "increasing": 0.0}
    for s in rounded:
        counts[s.trend] += 1
        total_dur[s.trend] += max(0.0, s.end_s - s.start_s)

    dominant = max(total_dur.items(), key=lambda kv: kv[1])[0]
    n_trends = len(rounded)
    # "trend changes" as number of boundaries between segments
    n_changes = max(0, n_trends - 1)

    lines: List[str] = []

    if mode in ("trend+summary", "trend_only"):
        for s in rounded:
            lines.append(f"{int(s.start_s)} seconds to {int(s.end_s)} seconds: {s.trend}")
        # counts lines
        lines.append(f"Number of stable trends: {counts['stable']}")
        lines.append(f"Number of decreasing trends: {counts['decreasing']}")
        lines.append(f"Number of increasing trends: {counts['increasing']}")

    if mode in ("trend+summary", "summary_only"):
        # narrative summary paragraph (SensorLLM Table 8 style)
        # NOTE: your pasted sample has slightly inconsistent "3 separate trends" vs actual segments.
        # Here we keep it consistent.
        para = []
        para.append(
            f"The sensor data represents readings taken from a {sensor_desc} "
            f"between {int(rounded[0].start_s)} and {int(rounded[-1].end_s)} seconds."
        )
        para.append(
            f"Analysis reveals {n_trends} separate trend segments within the data, "
            f"undergoing a cumulative total of {n_changes} shifts in direction."
        )
        # ordered mention: decreasing/increasing/stable totals (like your example)
        para.append(
            f"Encapsulating the outcomes, the data’s decreasing trend stretched across a total time of "
            f"{int(total_dur['decreasing'])} seconds, came after an increasing pattern observed over "
            f"{int(total_dur['increasing'])} seconds, and a stable trend for "
            f"{int(total_dur['stable'])} seconds in total."
        )
        para.append(f"The dominant trend is {dominant}.")
        lines.append(" ".join(para))

    return "\n".join(lines)

def build_sensorllm_style_answer_int(
    series,
    series_times=None,
    time_quantum_s=1.0,
    sensor_desc="state sensor",
):
    x = np.asarray(series)
    # if x has nan
    if np.isnan(x).any():
        mask = ~np.isnan(x)
        if mask.any():
            x[~mask] = x[mask][0]
        else:
            x[:] = 0
    n = len(x)
    t = np.asarray(series_times)

    # segment (run-length encoding)
    segments = []
    start = 0
    for i in range(1, n):
        if x[i] != x[start]:
            segments.append((start, i, x[start]))
            start = i
    segments.append((start, n, x[start]))

    # segment text
    lines = []
    for s, e, v in segments:
        t0, t1 = int(t[s]), int(t[e-1]+time_quantum_s)
        lines.append(f"{t0} seconds to {t1} seconds: value={int(v)}")

    # summary
    toggles = len(segments) - 1
    values, counts = np.unique(x, return_counts=True)
    dominant = values[np.argmax(counts)]
    dominant_ratio = counts.max() / n

    summary = [
        f"Number of toggles: {toggles}",
        f"Dominant value: {int(dominant)} ({dominant_ratio:.2%})",
    ]

    narrative = (
        f"The sensor data represents readings taken from a {sensor_desc} "
        f"between {int(t[0])} and {int(t[-1] + time_quantum_s)} seconds. "
        f"The state changed {toggles} time(s) and was mostly value {int(dominant)}."
    )

    return "\n".join(lines + summary + [narrative])

if __name__ == "__main__":

    # real sensor data from SWaT (after loading CSV)
    attack_csv = pd.read_csv("./data/SWaT/attack.csv")[COLS]
    normal_csv = pd.read_csv("./data/SWaT/normal.csv")[COLS]

    normal_stride = 30
    attack_stride = 15
    X_list, Y_list, Prompt_list = [], [], []

    # normal data에서 윈도우 단위로 슬라이딩하면서 센서 시계열을 분석하여 트렌드와 요약을 생성
    len_normal = len(normal_csv)
    for start in tqdm(range(0, len_normal - window_size + 1, normal_stride)):
        window_df = normal_csv.iloc[start:start + window_size]
        x = window_df[COLS].to_numpy(dtype=np.float32)  # (61, 5)
        answers = {}
        for col in COLS:
            series = window_df[col].to_numpy(dtype=np.float32)
            times = np.arange(len(series), dtype=np.float32) * 1.0  # 1s step (샘플 간 시간 간격)
            answer = build_sensorllm_style_answer_int(
                series=series,
                series_times=times,
                time_quantum_s=1.0,
                sensor_desc=f"{col} sensor",
            )if col.startswith("P") or col.startswith("MV") else build_sensorllm_style_answer_float(
                series=series,
                series_times=times,
                time_quantum_s=1.0,
                smooth_win=1,
                min_seg_s=5.0,
                stable_eps_quantile=0.10,
                slope_tau_quantile=0.50,
                sensor_desc=f"{col} sensor",
            )
            answers[col] = answer
            print(answer)
        X_list.append(x)
        Y_list.append(0)          # normal
        Prompt_list.append(answers)

    # attack data에서 윈도우 단위로 슬라이딩하면서 센서 시계열을 분석하여 트렌드와 요약을 생성
    len_attack = len(attack_csv)
    for start in tqdm(range(0, len_attack - window_size + 1, attack_stride)):
        window_df = attack_csv.iloc[start:start + window_size]
        x = window_df[COLS].to_numpy(dtype=np.float32)  # (61, 5)
        answers = {}
        for col in COLS:
            series = window_df[col].to_numpy(dtype=np.float32)
            times = np.arange(len(series), dtype=np.float32) * 1.0  # 1s step (샘플 간 시간 간격)
            answer = build_sensorllm_style_answer_int(
                series=series,
                series_times=times,
                time_quantum_s=1.0,
                sensor_desc=f"{col} sensor",
            ) if col.startswith("P") or col.startswith("MV") else build_sensorllm_style_answer_float(
                series=series,
                series_times=times,
                time_quantum_s=1.0,
                smooth_win=1,
                min_seg_s=5.0,
                stable_eps_quantile=0.10,
                slope_tau_quantile=0.50,
                sensor_desc=f"{col} sensor",
            )

            answers[col] = answer
        X_list.append(x)
        Y_list.append(1)          # abnormal
        Prompt_list.append(answers)

    # numpy arrays
    X = np.stack(X_list, axis=0)        # (N, T, C)
    Y = np.array(Y_list, dtype=np.int64)
    np.savez_compressed("./data/SWaT/inputs.npz", X=X,Y=Y)

    # prompts.jsonl
    with open("./data/SWaT/prompts.jsonl", "w", encoding="utf-8") as f:
        for i, p in enumerate(Prompt_list):
            obj = {"id": i}
            for col in COLS:
                obj[col] = p[col]
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")