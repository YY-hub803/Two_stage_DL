import os
import json
import numpy as np
import pandas as pd
import torch
import General_utils
import model
import glob

def load_data(dir_x,dir_c,num_sites,date_range,train_stats):
    date_length = len(date_range)

    # ------------------ 1. 加载新流域数据 ------------------
    print("[*] 正在加载新流域数据...")
    x_new = General_utils.load_timeseries(dir_x, num_sites, date_length)
    c_new = General_utils.load_attribute(dir_c)

    # ------------------ 2. 使用【训练集】的参数进行标准化 ------------------
    print("[*] 正在应用训练集统计特征进行标准化...")

    x_processed = x_new.copy()
    log_indices = list(range(x_new.shape[2]))
    for idx in log_indices:
        x_processed[:, :, idx] = np.log1p(x_processed[:, :, idx])

    # 核心：使用训练集的均值和标准差！
    x_norm_new = (x_processed - train_stats['x_mean']) / train_stats['x_std']
    c_norm_new = (c_new - train_stats['c_mean']) / train_stats['c_std']
    x_norm_new = np.nan_to_num(x_norm_new, nan=0.0)
    c_norm_new = np.nan_to_num(c_norm_new, nan=0.0)
    # ------------------ 3. 构建时间嵌入与特征拼接 ------------------
    date_processing = General_utils.Time_emb(date_range,train_stats=train_stats,date_ms=True)
    date_array = np.expand_dims(date_processing.values, axis=0)
    date_emb = np.repeat(date_array, num_sites, axis=0)

    c_repeated = np.repeat(c_norm_new[:, np.newaxis, :], date_length, axis=1)

    x_input_full = np.concatenate([x_norm_new,date_emb, c_repeated], axis=2)


    return x_input_full

def inference(model,input,num_sites,sites_ID,history_len,date_range,train_stats,target_names,saveFolder,device):
    model_name = model.__class__.__name__
    out_dim = len(target_names)
    date_length = len(date_range)

    print(f"[*] 开始滑窗推理 (窗口={history_len})...")
    prediction_sum = np.zeros((num_sites, date_length, out_dim))
    prediction_counts = np.zeros((num_sites, date_length, out_dim))

    total_steps = date_length - history_len + 1

    time_batch_size = 270  # 时间维度的 batch 大小，可根据显存调整

    model.eval()
    with torch.no_grad():
        for t_idx in range(0, total_steps, time_batch_size):
            t_end_idx = min(t_idx + time_batch_size, total_steps)
            windows = []
            t_starts = []

            for t in range(t_idx,t_end_idx):
                windows.append(input[:,t:t+history_len,:])
                t_starts.append(t)

            x_tensor = torch.tensor(np.array(windows), dtype=torch.float32).to(device)
            B_time = len(t_starts)
            x_tensor = x_tensor.view(B_time * num_sites, history_len, -1)
            if hasattr(model, 'attn_block'):
                preds, _ = model(x_tensor, return_attn=True)
            else:
                preds = model(x_tensor)


            preds = preds.view(B_time, num_sites, history_len, out_dim).cpu().numpy()
            for i, t in enumerate(t_starts):
                prediction_sum[:, t:t + history_len, :] += preds[i]
                prediction_counts[:, t:t + history_len, :] += 1

    prediction_counts[prediction_counts == 0] = 1
    final_outputs_norm = prediction_sum / prediction_counts
    # ------------------ 5. 反标准化与保存 ------------------
    print("[*] 反标准化并输出预测结果...")
    site_names = sites_ID["P_nm"].values if isinstance(sites_ID, pd.DataFrame) else sites_ID

    results_df = {}
    for i, var_name in enumerate(target_names):
        pred_raw = final_outputs_norm[:, :, i]

        # 获取训练集的 Target 参数
        cur_std = np.array(train_stats['y_std']).flat[i]
        cur_mean = np.array(train_stats['y_mean']).flat[i]
        # 反标准化 -> 反 Log
        pred_inv = pred_raw * cur_std + cur_mean
        pred_final = np.expm1(pred_inv)

        df_pred = pd.DataFrame(pred_final, index=site_names).T
        results_df[var_name] = df_pred

        if saveFolder:
            filePath = saveFolder + '/Inference' + f"{model_name}" + f'_{var_name}' + '.csv'
            if os.path.exists(filePath):
                os.remove(filePath)
            df_pred.to_csv(filePath, index=False)
    print("[*] 预测完成！")

    return results_df





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hyper_params = {
    "epoch_run": 500,
    "hidden_size": 512,
    'history_len': 365,
    "batch_size":128,
    "drop_rate": 0.3,
    "warmup_epochs":10,
    "base_lr":1e-4,
    "BACKEND":"MoE_LSTM", # select model    LSTMModel/CNNLSTMmodel/ATTLSTMModel/ATCLSTMModel
    "lossFun":'RMSE'
}


MODEL_FACTORY = {
    "LSTMModel": model.LSTMModel,
    "ATTLSTMModel": model.ATTLSTMModel,
    "CNNLSTMmodel":model.CNNLSTMmodel,
    "ATCLSTMModel":model.ATCLSTMModel,
    "MoE_LSTM":model.MoE_LSTM,
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
dir_proj = f"sangamon"
work_path = os.getcwd()
dir_input = os.path.join(work_path, dir_proj)
dir_output = os.path.join("inference",dir_model)
model_path = os.path.join("OutPut",dir_model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(dir_output):
    # 创建文件夹，如果有必要会创建中间目录
    os.makedirs(dir_output, exist_ok=True)
    print(f"成功创建模型输出文件夹: {dir_output}")
else:
    print(f"模型输出文件夹已存在: {dir_output}")

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

Target_Name = list(dir_y.keys())

with open('OutPut/train_stats.json', 'r', encoding='utf-8') as f:
    train_stats = json.load(f)
print('  ------------------------loading sites_info ------------------------------')
sites_ID= pd.read_csv(os.path.join(dir_input,"points_info.csv"))
coords = sites_ID.iloc[:,2:4].values

num_sites = len(sites_ID)
full_date = pd.date_range('1980-01-01', '1997-12-31')
date_length = len(full_date)

x_in = load_data(dir_x,dir_c,num_sites,full_date,train_stats)

model_files = glob.glob(os.path.join(model_path, "*.pt"))
if not model_files:
    raise FileNotFoundError("未能找到训练保存的模型文件，请检查 train_G 是否成功保存。")

latest_model_path = max(model_files, key=os.path.getmtime)
print(f">>> 加载原始模型: {latest_model_path}")
model_raw = torch.load(latest_model_path)
y_out = inference(model_raw,x_in,num_sites,sites_ID,hyper_params['history_len'],full_date,train_stats,Target_Name,dir_output,device)