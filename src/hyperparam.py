data_window_size = 90
data_stride = 15
data_len = 64
normal_data_cut = 10
attack_data_cut = 1

epoch_num = 200
train_epoch_num = 1
model_name = "DilatedCNN" # "GPT" or "LSTM" or "DilatedCNN"

cpu_num = 4
device = "cuda"
batch_size = 8
lr = 1e-4
weight_decay = 1e-4
d_model = 64
n_heads = 4
n_layers = 4
dropout = 0.0
pooling_layer_num = 7
masking_len = 20
masking_ratio = 0.3
