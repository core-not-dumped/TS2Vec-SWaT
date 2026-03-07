from tqdm import tqdm
import time
from thop import profile
from model.inputProjection import *
from model.pooling import *
from model.customGPT import *
from model.customLSTM import *
from model.customDilatedCNN import *
from model.loss import *
from data.dataset import *
from data.augmentation import *
from src.hyperparam import *
from src.report import *

def write_and_print(f, msg):
    print(msg)
    f.write(msg + "\n")
    f.flush()

train_data_loader_general_hyperparam = {
    "batch_size": batch_size,
    "shuffle": True,
    "num_workers": cpu_num,
}
test_data_loader_general_hyperparam = {
    "batch_size": 1,
    "shuffle": True,
    "num_workers": cpu_num,
}
normal_train_dataset = SWaTWindowDataset([f"./data/SWaT_processed/normal_{data_len * 3}_{i}.npz" for i in range(0, 8)])
normal_test_dataset = SWaTWindowDataset([f"./data/SWaT_processed/normal_{data_len * 3}_{i}.npz" for i in range(8, 10)])
normal_train_dataloader = torch.utils.data.DataLoader(normal_train_dataset, **train_data_loader_general_hyperparam)
normal_test_dataloader = torch.utils.data.DataLoader(normal_test_dataset, **test_data_loader_general_hyperparam)

channel_num = normal_train_dataset.x.shape[-1]
model = CustomDilatedCNN(d_model=d_model, n_layers=n_layers, kernel_size=3, dropout=dropout).to(device)
model.load_state_dict(torch.load(f"./model/{model_name}/199.pt", map_location=device, weights_only=True))
proj_layer = InputProjection_W_TimeSensorMasking(channel_num, d_model, time_masking_ratio=time_masking_ratio, sensor_masking_ratio=sensor_masking_ratio).to(device)
proj_layer.load_state_dict(torch.load(f"./model/{model_name}/199_proj.pt", map_location=device, weights_only=True))
pooling_layer = TS2VecMaxPooling(pooling_layer_num).to(device)

input = torch.randn(64, data_len, channel_num).to(device)
proj_flops, proj_params = profile(proj_layer, inputs=(input,))
proj_out = proj_layer(input)
print("FLOPs:", proj_flops)
print("Params:", proj_params)

model_flops, model_params = profile(model, inputs=(proj_out,))
model_out = model(proj_out)
print("FLOPs:", model_flops)
print("Params:", model_params)

pooling_flops, pooling_params = profile(pooling_layer, inputs=(model_out,))
pooling_out = pooling_layer(model_out)
print("FLOPs:", pooling_flops)
print("Params:", pooling_params)

print(f"Total FLOPs: {proj_flops + model_flops + pooling_flops}")
print(f"Total Params: {proj_params + model_params + pooling_params}")

input = torch.randn(64, data_len, channel_num).to(device)
torch.cuda.synchronize()
start_time = time.time()
proj_out = proj_layer(input)
model_out = model(proj_out)
pooling_out = pooling_layer(model_out)
torch.cuda.synchronize()
end_time = time.time()
print(f"Inference time: {end_time - start_time:.4f} seconds")
