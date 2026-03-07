from datetime import datetime
import os
import wandb

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
from src.callback import *

def write_and_print(f, msg):
    print(msg)
    f.write(msg + "\n")
    f.flush()

data_loader_general_hyperparam = {
    "batch_size": batch_size,
    "shuffle": True,
    "num_workers": cpu_num,
}
normal_train_dataset = SWaTWindowDataset([f"./data/SWaT_processed/normal_{i}.npz" for i in range(0, 8)])
normal_train_dataloader = torch.utils.data.DataLoader(normal_train_dataset,**data_loader_general_hyperparam)
normal_test_dataset = SWaTWindowDataset([f"./data/SWaT_processed/normal_{i}.npz" for i in range(8, 10)])
normal_test_dataloader = torch.utils.data.DataLoader(normal_test_dataset, **data_loader_general_hyperparam)
attack_dataset = SWaTWindowDataset(["./data/SWaT_processed/attack_0.npz"])
attack_dataloader = torch.utils.data.DataLoader(attack_dataset, **data_loader_general_hyperparam)

channel_num = normal_train_dataset.x.shape[-1]
# proj_layer = InputProjection_W_TimestampMasking(channel_num, d_model, mask_prob=time_masking_ratio).to(device)
proj_layer = InputProjection_W_TimeSensorMasking(channel_num, d_model, time_masking_ratio=time_masking_ratio, sensor_masking_ratio=sensor_masking_ratio).to(device)
if model_name == "GPT":
    cfg = customGPTConfig(
        in_channels=channel_num,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout
    )
    model = CustomGPT(cfg).to(device)
elif model_name == "LSTM":
    model = CustomLSTM(d_model=d_model, n_layers=2, dropout=dropout).to(device)
elif model_name == "DilatedCNN":
    model = CustomDilatedCNN(d_model=d_model, n_layers=n_layers, kernel_size=3, dropout=dropout).to(device)
pooling_layer = TS2VecMaxPooling(pooling_layer_num).to(device)
params = list(model.parameters()) + list(proj_layer.parameters())
optimizers = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
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

for epoch in range(epoch_num):
    print(f"Epoch {epoch+1}/{epoch_num}")
    
    # collapse/spread: pairwise cosine stats
    sim_mean_list, sim_std_list = [], []
    # augmentation invariance: view distance stats
    aug_dist_mean_list, aug_dist_p90_list = [], []
    # embedding norm stats (pre-normalize)
    norm1_mean_list, norm2_mean_list = [], []

    # training
    print("Training...")
    for train_epoch in range(train_epoch_num):
        losses = []
        model.train()
        proj_layer.train()
        for x, y, ts in tqdm(normal_train_dataloader):
            x = x.to(device) # (B, T, C)

            # data augmentation
            x1, x2 = augment_view_return2(x, data_len) # (B, data_len, C)

            # pooling + timestamp masking
            x1 = proj_layer(x1, no_mask=False) # (B, data_len, d_model)
            x2 = proj_layer(x2, no_mask=False) # (B, data_len, d_model)

            # Dilated Convolution (Transformer, LSTM, CNN..., main model)
            out1 = model(x1) # (B, data_len, d_model)
            out2 = model(x2) # (B, data_len, d_model)

            # pooling layer
            outs1 = pooling_layer(out1)
            outs2 = pooling_layer(out2)

            # loss backward
            loss = criterion(outs1, outs2)
            optimizers.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)

            # gradient 확인
            '''
            total_norm = 0
            for p in list(model.parameters()) + list(proj_layer.parameters()):
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"model grad_norm: {total_norm:.2f}")
            '''
            
            optimizers.step()

            losses.append(loss.item())
            with torch.no_grad():
                # embedding norm (폭발/수축 감지)
                n1 = outs1[-1][:,:,-1].norm(dim=-1)  # (B,)
                n2 = outs2[-1][:,:,-1].norm(dim=-1)
                norm1_mean_list.append(n1.mean().item())
                norm2_mean_list.append(n2.mean().item())

                # augmentation invariance (view1 vs view2)
                z1n = F.normalize(outs1[-1][:,:,-1], dim=-1)
                z2n = F.normalize(outs2[-1][:,:,-1], dim=-1)
                aug_dist = 1.0 - (z1n * z2n).sum(dim=-1)  # cosine distance, (B,)
                aug_dist_mean_list.append(aug_dist.mean().item())
                aug_dist_p90_list.append(torch.quantile(aug_dist, 0.9).item())

                # representation spread / collapse proxy:
                # batch 내 pairwise cosine similarity 분포의 mean/std
                # (B,B) sim matrix (대각 포함)
                sim = z1n @ z1n.transpose(0, 1)
                sim_mean_list.append(sim.mean().item())
                sim_std_list.append(sim.std().item())

        # epoch summary
        loss_ep = float(np.mean(losses))

        sim_mean_ep = float(np.mean(sim_mean_list))
        sim_std_ep  = float(np.mean(sim_std_list))
        aug_mean_ep = float(np.mean(aug_dist_mean_list))
        aug_p90_ep  = float(np.mean(aug_dist_p90_list))
        n1_ep       = float(np.mean(norm1_mean_list))
        n2_ep       = float(np.mean(norm2_mean_list))

        print(
            f"TrainEpoch {train_epoch}\n"
            f"loss={loss_ep:.6f}\n"
            f"sim_mean={sim_mean_ep:.4f} sim_std={sim_std_ep:.4f}\n"
            f"aug_dist_mean={aug_mean_ep:.4f} aug_dist_p90={aug_p90_ep:.4f}\n"
            f"norm(z1)={n1_ep:.2f} norm(z2)={n2_ep:.2f}"
        )

    with torch.no_grad():
        print("Evaluating...")

        # score by lastmask
        scores_n_train, labels_n_train = score_by_learnable_masking_random(
            model, proj_layer, pooling_layer, normal_train_dataloader, device, masking_len=masking_len, progress=0.5,
        )
        scores_n, labels_n = score_by_learnable_masking_random(
            model, proj_layer, pooling_layer, normal_test_dataloader, device, masking_len=masking_len, progress=1.0,
        )
        scores_a, labels_a = score_by_learnable_masking_random(
            model, proj_layer, pooling_layer, attack_dataloader, device, masking_len=masking_len, progress=1.0,
        )

        def summarize_scores(name, scores, f):
            msg = (
                f"[{name}] n={len(scores)}  "
                f"mean={scores.mean():.4f}  "
                f"std={scores.std():.4f}  "
                f"p1={np.percentile(scores,1):.4f}  "
                f"p10={np.percentile(scores,10):.4f}  "
                f"p50={np.percentile(scores,50):.4f}  "
                f"p90={np.percentile(scores,90):.4f}  "
                f"p99={np.percentile(scores,99):.4f}"
            )
            write_and_print(f, msg)

        with open("./results.txt", "a") as f:
            write_and_print(f, f"Epoch {epoch}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            summarize_scores("normal_train", scores_n_train, f)
            summarize_scores("normal_test", scores_n, f)
            summarize_scores("attack_test", scores_a, f)

            # threshold 예시: normal_train(or normal_test)의 1%를 임계값으로
            thr = np.percentile(scores_n_train, 99)
            write_and_print(f, f"threshold(p99 of normal_train): {thr:.6f}")
            write_and_print(f, f"attack detection rate @thr: {(scores_a > thr).mean():.3f}")
            write_and_print(f, f"false positive rate @thr: {(scores_n > thr).mean():.3f}")
            write_and_print(f, f"top attack percentage @thr: {topk_percentage(scores_n, scores_a)[0]:.3f}")
            write_and_print(f, f"top normal percentage @thr: {topk_percentage(scores_n, scores_a)[1]:.3f}")

            os.makedirs(f"./model/{model_name}", exist_ok=True)
            torch.save(model.state_dict(), f"./model/{model_name}/{epoch}.pt")
            torch.save(proj_layer.state_dict(), f"./model/{model_name}/{epoch}_proj.pt")
            print(f"model saved to ./model/{model_name}/{epoch}.pt and ./model/{model_name}/{epoch}_proj.pt")
            write_and_print(f, "-" * 80)
        logger.log_val(
            threshold=thr,
            attack_detection_rate=(scores_a > thr).mean(),
            false_positive_rate=(scores_n > thr).mean(),
            top_attack_percentage=topk_percentage(scores_n, scores_a)[0],
            top_normal_percentage=topk_percentage(scores_n, scores_a)[1],
            loss_ep=loss_ep,
        )

wandb.finish()