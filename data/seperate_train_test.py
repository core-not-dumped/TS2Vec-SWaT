import os
import json
import numpy as np

def split_swat_npz_and_jsonl(
    npz_path: str,
    jsonl_path: str,
    out_dir: str,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) JSONL 로드
    with open(jsonl_path, "r", encoding="utf-8") as f:
        jsonl_lines = [line.rstrip("\n") for line in f if line.strip()]

    N_json = len(jsonl_lines)
    if N_json == 0:
        raise ValueError("prompts.jsonl is empty (no valid lines).")

    # 2) NPZ 로드 + N 확인
    npz = np.load(npz_path, allow_pickle=True)
    keys = list(npz.keys())

    # N을 결정: jsonl과 맞는 첫 번째 (N, ...) 배열을 찾아서 N로 사용
    N = None
    n_first_axis_keys = []
    for k in keys:
        arr = npz[k]
        if hasattr(arr, "shape") and len(arr.shape) >= 1 and arr.shape[0] == N_json:
            N = N_json
            n_first_axis_keys.append(k)

    if N is None:
        # jsonl과 같은 N인 배열이 하나도 없으면, npz에서 가장 그럴듯한 N 후보를 찾고 에러 메시지 제공
        candidates = []
        for k in keys:
            arr = npz[k]
            if hasattr(arr, "shape") and len(arr.shape) >= 1:
                candidates.append((k, arr.shape[0], arr.shape))
        raise ValueError(
            f"Could not find any arrays in NPZ whose first dimension matches JSONL lines.\n"
            f"JSONL lines: {N_json}\n"
            f"NPZ first-dim candidates: {candidates[:10]} (showing up to 10)\n"
            f"Check that inputs.npz and prompts.jsonl correspond to the same dataset ordering."
        )

    # 추가로: jsonl 라인 수와 N 일치 확인 (이미 N=N_json)
    if N != N_json:
        raise ValueError(f"N mismatch: npz N={N}, jsonl lines={N_json}")

    # 3) 인덱스 셔플 + 분할
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)

    test_size = int(round(N * test_ratio))
    test_idx = np.sort(idx[:test_size])
    train_idx = np.sort(idx[test_size:])

    # 4) JSONL 저장
    train_jsonl_path = os.path.join(out_dir, "train_prompts.jsonl")
    test_jsonl_path  = os.path.join(out_dir, "test_prompts.jsonl")

    with open(train_jsonl_path, "w", encoding="utf-8") as f:
        for i in train_idx:
            f.write(jsonl_lines[i] + "\n")

    with open(test_jsonl_path, "w", encoding="utf-8") as f:
        for i in test_idx:
            f.write(jsonl_lines[i] + "\n")

    # 5) NPZ 저장
    #    - first-dim이 N인 배열: train/test로 분할 저장
    #    - 그 외 배열: 그대로 복사 저장
    train_npz_path = os.path.join(out_dir, "train_inputs.npz")
    test_npz_path  = os.path.join(out_dir, "test_inputs.npz")

    train_dict = {}
    test_dict  = {}

    for k in keys:
        arr = npz[k]
        if hasattr(arr, "shape") and len(arr.shape) >= 1 and arr.shape[0] == N:
            # 샘플 축이 있는 배열은 분할
            train_dict[k] = arr[train_idx]
            test_dict[k]  = arr[test_idx]
        else:
            # 메타/설정 배열은 그대로 복사
            train_dict[k] = arr
            test_dict[k]  = arr

    np.savez_compressed(train_npz_path, **train_dict)
    np.savez_compressed(test_npz_path, **test_dict)

    # 6) 요약 출력
    print("Split complete!")
    print(f"Total N: {N}")
    print(f"Train: {len(train_idx)}  Test: {len(test_idx)}  (test_ratio={test_ratio})")
    print(f"Saved:")
    print(f"  {train_npz_path}")
    print(f"  {test_npz_path}")
    print(f"  {train_jsonl_path}")
    print(f"  {test_jsonl_path}")
    print(f"Arrays split (first-dim==N): {n_first_axis_keys}")

if __name__ == "__main__":
    split_swat_npz_and_jsonl(
        npz_path="./data/SWaT/inputs.npz",
        jsonl_path="./data/SWaT/prompts.jsonl",
        out_dir="./data/SWaT_split",
        test_ratio=0.2,
        seed=42,
    )