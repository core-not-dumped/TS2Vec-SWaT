import optuna
import torch
import numpy as np
import torch.nn.functional as F
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
from src.callback import *

# ---- (선택) 재현성/속도용 ----
def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_dataloaders(batch_size: int):
    data_loader_general_hyperparam = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": cpu_num,
    }
    normal_train_dataset = SWaTWindowDataset([f"./data/SWaT_processed/normal_{i}.npz" for i in range(0, 8)])
    normal_test_dataset  = SWaTWindowDataset([f"./data/SWaT_processed/normal_{i}.npz" for i in range(8, 10)])
    attack_dataset       = SWaTWindowDataset(["./data/SWaT_processed/attack_0.npz"])

    normal_train_dataloader = torch.utils.data.DataLoader(normal_train_dataset, **data_loader_general_hyperparam)
    normal_test_dataloader  = torch.utils.data.DataLoader(normal_test_dataset,  **data_loader_general_hyperparam)
    attack_dataloader       = torch.utils.data.DataLoader(attack_dataset,       **data_loader_general_hyperparam)

    channel_num = normal_train_dataset.x.shape[-1]
    return channel_num, normal_train_dataloader, normal_test_dataloader, attack_dataloader

def build_model_and_proj(
    channel_num: int,
    d_model: int,
    time_masking_ratio: float,
    sensor_masking_ratio: float,
):
    proj_layer = InputProjection_W_TimeSensorMasking(
        channel_num,
        d_model,
        time_masking_ratio=time_masking_ratio,
        sensor_masking_ratio=sensor_masking_ratio,
    ).to(device)

    if model_name == "GPT":
        cfg = customGPTConfig(
            in_channels=channel_num,
            d_model=d_model,
            n_heads=n_heads,      # 기존 전역값 사용
            n_layers=n_layers,    # 기존 전역값 사용
            dropout=dropout,
        )
        model = CustomGPT(cfg).to(device)
    elif model_name == "LSTM":
        model = CustomLSTM(d_model=d_model, n_layers=2, dropout=dropout).to(device)
    elif model_name == "DilatedCNN":
        model = CustomDilatedCNN(d_model=d_model, n_layers=6, kernel_size=3, dropout=dropout).to(device)
    else:
        raise ValueError(f"Unknown model_name={model_name}")

    pooling_layer = TS2VecMaxPooling(pooling_layer_num).to(device)
    return model, proj_layer, pooling_layer

def train_one_trial(
    trial,
    lr: float,
    batch_size: int,
    d_model: int,
    time_masking_ratio: float,
    sensor_masking_ratio: float,
):
    trial_epoch_num = 10
    trial_train_epoch_num = 1

    channel_num, normal_train_dl, normal_test_dl, attack_dl = build_dataloaders(batch_size)
    model, proj_layer, pooling_layer = build_model_and_proj(
        channel_num, d_model, time_masking_ratio, sensor_masking_ratio
    )

    optimizers = torch.optim.AdamW(list(model.parameters()) + list(proj_layer.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = hier_loss_ts2vec_dual

    wandb_config = {
        "model_name": model_name,
        "lr": lr,
        "batch_size": batch_size,
        "feature_dim": d_model,
        "mask_ratio_time": time_masking_ratio,
        "mask_ratio_sensor": sensor_masking_ratio,
    }
    wandb.init(
        project="ts2vec-anomaly",
        name=", ".join(f"{k}={v}" for k, v in wandb_config.items()),
        config=wandb_config,
        save_code=True,
    )
    logger = WandBLogger()

    model.train()
    proj_layer.train()
    for epoch in range(trial_epoch_num):
        for train_epoch in range(trial_train_epoch_num):
            for x, y, ts in tqdm(normal_train_dl):
                x = x.to(device)  # (B, T, C)

                x1, x2 = augment_view_return2(x, data_len)  # (B, data_len, C)
                x1 = proj_layer(x1, no_mask=False)
                x2 = proj_layer(x2, no_mask=False)

                out1 = model(x1)
                out2 = model(x2)

                outs1 = pooling_layer(out1)
                outs2 = pooling_layer(out2)

                loss = criterion(outs1, outs2)

                optimizers.zero_grad()
                loss.backward()
                optimizers.step()

        model.eval()
        proj_layer.eval()
        with torch.no_grad():
            scores_n_train, _ = score_by_learnable_masking_random(
                model, proj_layer, pooling_layer, normal_train_dl, device, masking_len=masking_len, progress=0.2
            )
            scores_n, _ = score_by_learnable_masking_random(
                model, proj_layer, pooling_layer, normal_test_dl, device, masking_len=masking_len, progress=1.0
            )
            scores_a, _ = score_by_learnable_masking_random(
                model, proj_layer, pooling_layer, attack_dl, device, masking_len=masking_len, progress=1.0
            )

            thr = np.percentile(scores_n_train, 99)
            attack_detection_rate = float((scores_a > thr).mean())
            false_positive_rate = float((scores_n > thr).mean())
            top_attack_percentage = topk_percentage(scores_n, scores_a)[0]
            top_normal_percentage = topk_percentage(scores_n, scores_a)[1]
        logger.log_val(
            threshold=thr,
            attack_detection_rate=attack_detection_rate,
            false_positive_rate=false_positive_rate,
            top_attack_percentage=top_attack_percentage,
            top_normal_percentage=top_normal_percentage,
        )

    # (선택) 보조 지표도 저장해두면 나중에 보기 편함
    trial.set_user_attr("thr", float(thr))
    trial.set_user_attr("fpr", false_positive_rate)
    wandb.finish()

    return attack_detection_rate  # maximize 대상

def objective(trial: optuna.Trial):
    # ---- search space ----
    lr = trial.suggest_float("lr", 3e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    time_masking_ratio = trial.suggest_float("time_masking_ratio", 0.05, 0.6)
    sensor_masking_ratio = trial.suggest_float("sensor_masking_ratio", 0.0, 0.3)

    return train_one_trial(
        trial,
        lr=lr,
        batch_size=batch_size,
        d_model=d_model,
        time_masking_ratio=time_masking_ratio,
        sensor_masking_ratio=sensor_masking_ratio,
    )

def run_optuna(n_trials: int = 30, seed: int = 42):
    seed_everything(seed)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="ts2vec_anomaly_opt",
    )
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print("\n===== OPTUNA DONE =====")
    print("Best value (attack_detection_rate):", study.best_value)
    print("Best params:", study.best_params)
    print("Best trial attrs:", study.best_trial.user_attrs)
    return study

study = run_optuna(n_trials=50, seed=42)
