data_window_size = 90
data_stride = 15
data_len = 64
normal_data_cut = 10
attack_data_cut = 1

epoch_num = 200
train_epoch_num = 1

cpu_num = 4
device = "cuda"
batch_size = 8
lr = 1e-3
weight_decay = 1e-5
use_amp = True
d_model = 64
n_heads = 4
n_layers = 4
dropout = 0.0
pooling_layer_num = 7
masking_len = data_len
masking_ratio = 0.3