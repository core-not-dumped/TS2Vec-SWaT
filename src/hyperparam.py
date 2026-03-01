data_window_size = 90
data_stride = 15
data_len = 60
normal_data_cut = 10
attack_data_cut = 1

epoch_num = 200
train_epoch_num = 1

cpu_num = 4
device = "cuda"
batch_size = 8
lr = 3e-4
weight_decay = 1e-5
use_amp = True
d_model = 128
n_heads = 4
n_layers = 4
dropout = 0.1
pooling_layer_num = 7
slide_len = 5
masking_len = data_len
masking_ratio = 0.3
tau = 0.2