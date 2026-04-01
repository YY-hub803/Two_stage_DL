import random
import json
import numpy as np
import pandas as pd
import torch
import train
import shutil
import model
import crit
import glob
import os
import General_utils
import Visualization as vis


def set_seeds(seed_value):
    """Set seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# set seeds
random_seed = 40
set_seeds(random_seed)

# set GPU
if torch.cuda.is_available():
    GPUid = 0
    torch.cuda.set_device(GPUid)


hyper_params = {
    "epoch_run": 500,
    "hidden_size": 512,
    'history_len': 365,
    "batch_size":430,
    "drop_rate": 0.3,
    "warmup_epochs":10,
    "base_lr":1e-4,
    "BACKEND":"MoE_LSTM", # select model    LSTMModel/CNNLSTMmodel/ATTLSTMModel/ATCLSTMModel/MoEAttLSTM
    "lossFun":'RMSE'
}


MODEL_FACTORY = {
    "LSTMModel": model.LSTMModel,
    "ATTLSTMModel": model.ATTLSTMModel,
    "CNNLSTMmodel":model.CNNLSTMmodel,
    "ATCLSTMModel":model.ATCLSTMModel,
    'MoE_LSTM':model.MoE_LSTM
}
Loss_FACTORY = {
    "MSE": crit.MSELoss,
    "NSE": crit.NSELoss,
    "RMSE": crit.RMSELoss,
    "MixLoss": crit.MixLoss,
    "WeightLoss":crit.WeightLoss
}



freq = '1D'
dir_model = freq+'_' + "%s_H%d_L%d_dr%.2f_E%d" % (
    hyper_params['BACKEND'],
    hyper_params['hidden_size'],
    hyper_params['history_len'],
    hyper_params['drop_rate'],
    hyper_params['epoch_run'],
)
# set input and output folders
dir_proj = f"data"
output_dir = "OutPut"
work_path = os.getcwd()
os.makedirs(output_dir, exist_ok=True)
dir_input = os.path.join(work_path, dir_proj)
dir_output = os.path.join(output_dir,dir_model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BACKEND= hyper_params["BACKEND"]

num_sites = 430
full_date = pd.date_range('1980-01-01', '2019-12-31')
full_date_train = pd.date_range('1980-01-01', '2009-12-31')
full_date_test = pd.date_range('2010-01-01', '2019-12-31')
date_length = len(full_date)

#------------------------------------- load data -----------------------------------------------------------------------
print("------------------------ load path ------------------------------")
dir_x = {
    "x_pre": os.path.join(dir_input, 'input_xforce_prcp.csv'),
    "x_swe": os.path.join(dir_input, 'input_xforce_swe.csv'),
    "x_dayl":os.path.join(dir_input, 'input_xforce_dayl.csv'),
    "x_tmax": os.path.join(dir_input, 'input_xforce_tmax.csv'),
    "x_tmin": os.path.join(dir_input, 'input_xforce_tmin.csv'),
    "x_vp": os.path.join(dir_input, 'input_xforce_vp.csv'),
}

dir_c = {
    "c_all": os.path.join(dir_input, 'input_c_all.csv'),
}

dir_y = {
    # "Flux": os.path.join(dir_input, 'input_yobs_Flux.csv'),
    "DIS": os.path.join(dir_input, 'input_yobs_Dis.csv'),
    "TP": os.path.join(dir_input, 'input_yobs_Tp.csv')
}


print("------------------------ load data ------------------------------")

print('  Loading X (Forcing)...')
x = General_utils.load_timeseries(dir_x, num_sites, date_length)

print('  Loading C (Static Attributes)...')
c = General_utils.load_attribute(dir_c)
c[np.where(np.isnan(c))] = 0

print('  Loading Y (Targets)...')
y = General_utils.load_timeseries(dir_y, num_sites, date_length)

print("------------------------ processing data ------------------------------")
# 划分训练站点
all_sites = np.arange(num_sites)
np.random.shuffle(all_sites)
train_ratio = 0.7
val_ratio = 0.2
num_train = int(num_sites * train_ratio)
num_val = int(num_sites * val_ratio)
train_sites = all_sites[:num_train]
val_sites = all_sites[num_train:num_train + num_val]
test_sites = all_sites[num_train + num_val:]
print(f"站点划分 -> 训练集: {len(train_sites)}个, 验证集: {len(val_sites)}个, 测试集: {len(test_sites)}个")

date_processing = General_utils.Time_emb(full_date)
date_array = date_processing.values
date_array_expanded = np.expand_dims(date_array, axis=0)
date_emb = np.repeat(date_array_expanded, num_sites, axis=0)

c_norm, c_mean,c_std = General_utils.preprocess_static_data(c, log_indices=None, train_indices=train_sites)

# list(range(x.shape[2]))
x_norm, x_mean, x_std = General_utils.preprocess_dynamic_data(x, log_indices=list(range(x.shape[2])),train_indices=train_sites)

x_norm = np.concatenate([x_norm,date_emb], axis=2)

y_norm, y_mean, y_std = General_utils.preprocess_dynamic_data(y, log_indices=list(range(y.shape[2])),train_indices=train_sites)


stats = {
    "c_mean": c_mean.tolist(),
    "c_std": c_std.tolist(),
    "x_mean": x_mean.tolist(),
    "x_std": x_std.tolist(),
    "y_mean": y_mean.tolist(),
    "y_std": y_std.tolist(),
}
json_path = os.path.join(output_dir, "train_stats.json")
with open(json_path, "w") as f:
    json.dump(stats, f, indent=4)

print('  ------------------------loading sites_info ------------------------------')
sites_ID= pd.read_csv(os.path.join(dir_input,"points_info.csv"))
coords = sites_ID.iloc[:,2:4].values

print('  loading date split ...\n')
date_split = pd.read_csv(os.path.join(dir_input, 'TP_splitting.csv'))
date_split[['S_Training', 'E_Training', 'S_Testing', 'E_Testing']] = date_split[['S_Training', 'E_Training', 'S_Testing', 'E_Testing']].apply(pd.to_datetime)
print('output location:', dir_output, '\n')

nx = x_norm.shape[-1]+ c.shape[-1]
ny = y_norm.shape[-1]
nc = c.shape[-1]
if BACKEND in ("MoE_LSTM"):
    model = MODEL_FACTORY[BACKEND](
        nx, ny,nc,
        hyper_params['hidden_size'],
        hyper_params['drop_rate'])
else:
    model = MODEL_FACTORY[BACKEND](
        nx, ny,
        hyper_params['hidden_size'],
        hyper_params['drop_rate'])

print(f"模型参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


lossFun = Loss_FACTORY[hyper_params['lossFun']]()


model_test = train.train(
    model,x_norm, y_norm,c_norm,
    date_split,
    lossFun,
    hyper_params['epoch_run'],
    device, dir_output,
    hyper_params['warmup_epochs'],
    hyper_params['base_lr'],
    hyper_params['batch_size'],
    hyper_params['history_len'],
    train_sites=train_sites,
    val_sites=val_sites)


model_files = glob.glob(os.path.join(dir_output, "*.pt"))
if not model_files:
    raise FileNotFoundError("未能找到训练保存的模型文件，请检查 train_G 是否成功保存。")
# 按照文件修改时间排序，获取最新的模型

latest_model_path = max(model_files, key=os.path.getmtime)

print(f">>> 加载原始模型进行插补: {latest_model_path}")
model_raw = torch.load(latest_model_path)
Target_Name = list(dir_y.keys())

# --- 新增：仅提取测试集站点的数据 ---
x_test = x_norm[test_sites, :, :]
y_test = y_norm[test_sites, :, :]
c_test = c_norm[test_sites, :]
sites_ID_test = sites_ID.iloc[test_sites].copy()

y_out, y_true = train.Interpolation(
    model_raw, x_norm,y_norm,c_norm,
    y_mean, y_std, sites_ID, dir_output, Target_Name,device,
    hyper_params['history_len'])

# ------------------------ 可视化部分 ------------------------------

vis_folder = os.path.join(dir_output, 'visualization')

if not os.path.exists(vis_folder):
    # 创建文件夹，如果有必要会创建中间目录
    os.makedirs(vis_folder, exist_ok=True)
    print(f"成功创建模型输出文件夹: {vis_folder}")
else:
    print(f"模型输出文件夹已存在: {vis_folder}")
    shutil.rmtree(vis_folder, ignore_errors=True)
    os.makedirs(vis_folder, exist_ok=True)

if 'y_out' in locals():
    print("------------------------ 生成可视化图表 ------------------------------")
    vis_mapping = {
        "Flux": lambda: vis.vis_filled(y_true['Flux'], y_out['Flux'], full_date, vis_folder, "Flux"),
        "DIS": lambda: vis.vis_filled(y_true['DIS'], y_out['DIS'], full_date, vis_folder, "DIS"),
        "TP": lambda: vis.vis_filled(y_true['TP'], y_out['TP'], full_date, vis_folder, "TP")
    }
    for var_name, vis_func in vis_mapping.items():
        if var_name in Target_Name:
            vis_func()
            print(f"已执行 {var_name} 的可视化，保存至 {vis_folder}")





