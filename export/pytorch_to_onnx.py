import torch
import os

from model.inputProjection import *
from model.pooling import *
from model.customGPT import *
from model.customLSTM import *
from model.customDilatedCNN import *
from src.hyperparam import *
from data.dataset import *

train_data_loader_general_hyperparam = {
    "batch_size": batch_size,
    "shuffle": True,
    "num_workers": cpu_num,
}
normal_train_dataset = SWaTWindowDataset([f"./data/SWaT_processed/normal_{data_len * 3}_{i}.npz" for i in range(1, 8)])
normal_train_dataloader = torch.utils.data.DataLoader(normal_train_dataset, **train_data_loader_general_hyperparam)

channel_num = normal_train_dataset.x.shape[-1]

proj_layer = InputProjection_W_TimeSensorMasking(channel_num, d_model, time_masking_ratio=time_masking_ratio, sensor_masking_ratio=sensor_masking_ratio).to(device)
no_mask_proj_layer = no_mask_wrapper(proj_layer)
use_mask_proj_layer = use_mask_wrapper(proj_layer)
model = CustomDilatedCNN(d_model=d_model, n_layers=n_layers, kernel_size=3, dropout=dropout).to(device)
pooling_layer = TS2VecMaxPooling(pooling_layer_num)
proj_layer.load_state_dict(torch.load(f"model/{model_name}/199_proj.pt", map_location=device, weights_only=True))
model.load_state_dict(torch.load(f"model/{model_name}/199.pt", map_location=device, weights_only=True))
proj_layer.eval()
model.eval()

model_save_dir = f"export/onnx_test/models/"
os.makedirs(model_save_dir, exist_ok=True)

proj_dummy_input = torch.randn(1, data_len, channel_num).to(device)
torch.onnx.export(
    no_mask_proj_layer,
    proj_dummy_input,
    f"{model_save_dir}/no_mask_proj.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)
torch.onnx.export(
    use_mask_proj_layer,
    proj_dummy_input,
    f"{model_save_dir}/use_mask_proj.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)

model_dummy_input = torch.randn(1, data_len, d_model).to(device)
torch.onnx.export(
    model,
    model_dummy_input,
    f"{model_save_dir}/{model_name}.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)

pooling_dummy_input = torch.randn(1, data_len, d_model).to(device)
torch.onnx.export(
    pooling_layer,
    pooling_dummy_input,
    f"{model_save_dir}/hier_maxpooling.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)