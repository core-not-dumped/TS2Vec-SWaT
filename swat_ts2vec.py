from datetime import datetime

from tqdm import tqdm

from model.customGPT import *
from model.loss import *
from data.dataset import *
from data.augmentation import *
from src.hyperparam import *

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
cfg = customGPTConfig(
    in_channels=channel_num,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    dropout=dropout
)
model = CustomGPT(cfg).to(device)
pooling_layer = TS2VecMaxPooling(pooling_layer_num).to(device)
optimizers = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = hier_loss_ts2vec

for epoch in range(epoch_num):
    print(f"Epoch {epoch+1}/{epoch_num}")
    
    # training
    print("Training...")
    for train_epoch in range(train_epoch_num):
        losses = []
        model.train()
        for x, y, ts in tqdm(normal_train_dataloader):
            x = x.to(device)

            # data augmentation
            x1, x2 = augment_view_return2(x, data_len)

            # encoder 2번 forward (가중치 공유)
            out1 = model(x1)
            out2 = model(x2)

            outs1 = pooling_layer(out1)
            outs2 = pooling_layer(out2)

            loss = criterion(outs1, outs2, tau)

            optimizers.zero_grad()
            loss.backward()
            optimizers.step()

            losses.append(loss.item())
        print(f"Epoch {epoch} - Loss: {np.mean(losses)}")

    with torch.no_grad():
        print("Evaluating...")
        # centroid는 normal_train
        #mu = compute_centroid(model, pooling_layer, normal_train_dataloader, device)

        # normal_test, attack_test 점수
        #scores_n, labels_n = score_by_centroid(model, pooling_layer, normal_test_dataloader, mu, device)
        #scores_a, labels_a = score_by_centroid(model, pooling_layer, attack_dataloader, mu, device)

        # score by lastmask
        scores_n, labels_n = score_by_masking(
            model, pooling_layer, normal_test_dataloader, device, masking_len=masking_len,
            mask_value=0.0,
        )

        scores_a, labels_a = score_by_masking(
            model, pooling_layer, attack_dataloader, device, masking_len=masking_len,
            mask_value=0.0,
        )
        # score by sliding
        '''
        scores_n, labels_n = score_by_sliding(
            model, pooling_layer, normal_test_dataloader, device, slide_len=slide_len,
            mask_value=0.0,
        )

        scores_a, labels_a = score_by_sliding(
            model, pooling_layer, attack_dataloader, device, slide_len=slide_len,
            mask_value=0.0,
        )
        '''

        def summarize_scores(name, scores, f):
            msg = (
                f"[{name}] n={len(scores)}  "
                f"mean={scores.mean():.4f}  "
                f"std={scores.std():.4f}  "
                f"p50={np.percentile(scores,50):.4f}  "
                f"p90={np.percentile(scores,90):.4f}  "
                f"p99={np.percentile(scores,99):.4f}"
            )
            write_and_print(f, msg)

        with open("./results.txt", "a") as f:
            write_and_print(f, f"Epoch {epoch}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            summarize_scores("normal_test", scores_n, f)
            summarize_scores("attack_test", scores_a, f)

            # 합쳐서 AUROC/PR
            from sklearn.metrics import roc_auc_score, average_precision_score
            scores_all = np.concatenate([scores_n, scores_a])
            labels_all = np.concatenate([labels_n, labels_a])  # attack=1이어야 함
            write_and_print(f, f"AUROC: {roc_auc_score(labels_all, scores_all):.6f}")
            write_and_print(f, f"AUPRC: {average_precision_score(labels_all, scores_all):.6f}")

            # threshold 예시: normal_train(or normal_test)의 99%를 임계값으로
            thr = np.percentile(scores_n, 99)
            write_and_print(f, f"threshold(p99 of normal_test): {thr:.6f}")
            write_and_print(f, f"attack detection rate @thr: {(scores_a > thr).mean():.6f}")
            write_and_print(f, f"false positive rate @thr: {(scores_n > thr).mean():.6f}")

            torch.save(model.state_dict(), f"./model/customGPT/{epoch}.pt")