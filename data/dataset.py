import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.hyperparam import *

class SWaTWindowDataset(Dataset):
    """
    Returns:
      x: FloatTensor (T, C)
      y: LongTensor scalar
      answers: dict[str, str]  # per-sensor summaries
    """
    def __init__(self, npz_path: str, jsonl_path: str, cols=COLS):
        self.cols = list(cols)

        arr = np.load(npz_path)
        # X, Y 키 사용
        self.X = arr["X"].astype(np.float32)  # (N, T, C)
        self.Y = arr["Y"].astype(np.int64)    # (N,)

        # --- prompts.jsonl: 순서 기준으로 로드 ---
        self.answers = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.answers.append({c: obj[c] for c in self.cols})

        # sanity check
        if any(a is None for a in self.answers):
            missing = sum(a is None for a in self.answers)
            raise ValueError(f"prompts.jsonl에 누락된 id가 있습니다. missing={missing}")

        if len(self.X) != len(self.Y) or len(self.Y) != len(self.answers):
            raise ValueError("X/Y/prompts 길이가 서로 다릅니다.")

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])         # (T, C), float32
        y = torch.tensor(self.Y[idx], dtype=torch.long)  # scalar
        answers = self.answers[idx]               # dict[str,str]
        return x, y, answers


def collate_keep_answers(batch):
    """
    batch: list[(x, y, answers)]
    Returns:
      X: FloatTensor (B, T, C)
      Y: LongTensor (B,)
      answers_list: list[dict[str,str]]  # 배치별로 그대로 유지
    """
    xs, ys, answers = zip(*batch)
    X = torch.stack(xs, dim=0)
    Y = torch.stack(ys, dim=0)
    return X, Y, list(answers)


# dataloader 테스트
if __name__ == "__main__":
    ds = SWaTWindowDataset(
        npz_path="./data/SWaT/inputs.npz",
        jsonl_path="./data/SWaT/prompts.jsonl",
    )

    dl = DataLoader(
        ds,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_keep_answers,
        pin_memory=True,
    )

    X, Y, answers_list = next(iter(dl))
    print("X:", X.shape, X.dtype)        # (B, 61, 5) torch.float32
    print("Y:", Y.shape, Y.dtype)        # (B,) torch.int64
    print("answers[0] keys:", answers_list[0].keys())
    print("answers[0]['LIT101']:", answers_list[0]["LIT101"][:200], "...")