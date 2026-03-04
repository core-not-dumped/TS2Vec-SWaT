data_window_size = 192
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
lr = 7e-5
weight_decay = 1e-4
d_model = 64
n_heads = 4
n_layers = 6
dropout = 0.1
pooling_layer_num = 7
report_masking_len = 5
masking_len = 20
anomaly_ratio = 0.3
time_masking_ratio = 0.5
sensor_masking_ratio = 0.2
grad_clip = 20