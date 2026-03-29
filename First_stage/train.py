import numpy as np
import pandas as pd

from sklearn.metrics import r2_score,root_mean_squared_error,mean_squared_error
import hydroeval as he
from datetime import date
import os
import time
import torch
from torch.cuda.amp import GradScaler, autocast
import Visualization

scaler = GradScaler()

def saveModel(outFolder, model, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_ep' + str(epoch) + '.pt')
    torch.save(model, modelFile)


def loadModel(outFolder, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_ep' + str(epoch) + '.pt')
    model = torch.load(modelFile, weights_only=False)
    return model

def percentage_day(date_split):
    sites = date_split['ID'].unique()
    tempData = []
    for ind, s in enumerate(sites):
        S_training = date_split.loc[date_split['ID'] == s, 'S_Training']
        E_training = date_split.loc[date_split['ID'] == s, 'E_Training']
        d1 = date(S_training[ind].year, S_training[ind].month, S_training[ind].day)
        d2 = date(E_training[ind].year, E_training[ind].month, E_training[ind].day)
        delta = d2 - d1
        tempData.append(delta.days)
    temp = pd.Series(tempData)
    date_split['days_num'] = temp
    sumdays = np.sum(temp)
    tempPercent = []
    for s in sites:
        days = date_split.loc[date_split['ID'] == s, 'days_num'].values[0]
        tempPercent.append(days/sumdays)
    temp1 = pd.Series(tempPercent)
    date_split['day_percent'] = temp1
    return date_split

def selectSubset(x, iGrid, iT, rho, *, c=None, tupleOut=False):
    nx = x.shape[-1]
    nt = x.shape[1]
    if x.shape[0] == len(iGrid):
        iGrid = np.arange(0,len(iGrid))
        if nt <= rho:
            iT.fill(0)

    if iT is not None:
        batchSize = iGrid.shape[0]
        xTensor = torch.zeros([batchSize, rho,nx], requires_grad=False)
        for k in range(batchSize):
            temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]

            xTensor[k:k + 1,:,  :] = torch.from_numpy(temp)
    else:
        if len(x.shape) == 2:
            # Used for local calibration kernel
            # x = Ngrid * Ntime
            xTensor = torch.from_numpy(x[iGrid, :]).float()
        else:
            # Used for rho equal to the whole length of time series
            xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 0)).float()
            rho = xTensor.shape[0]
    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(
            np.reshape(c[iGrid, :], [batchSize, 1, nc]), rho, axis=1)
        cTensor = torch.from_numpy(temp).float()

        if (tupleOut):
            if torch.cuda.is_available():
                xTensor = xTensor.cuda()
                cTensor = cTensor.cuda()
            out = (xTensor, cTensor)
        else:
            out = torch.cat((xTensor, cTensor), 2)
    else:
        out = xTensor

    if torch.cuda.is_available() and type(out) is not tuple:
        out = out.cuda()
    return out

def randomIndex_percentage(ngrid, dimSubset, date_split_new):
    batchSize, rho = dimSubset

    subset_df = date_split_new.iloc[ngrid]
    probs = subset_df['day_percent'].values
    probs = probs / np.sum(probs)  # 重新归一化使其和为 1

    iGrid = np.random.choice(ngrid, size=batchSize, p=probs)
    iT = []
    for i in iGrid:
        nt = date_split_new.iloc[i]['days_num']
        T = np.random.randint(0, nt-rho, [1])[0]
        iT.append(T)
    return iGrid, iT


def randomIndex_percentage_test(nt_total, iGridTest, dimSubset, date_split_new):
    batchSize, rho = dimSubset
    #iGrid = np.random.choice(list(range(0, ngrid)), size=batchSize, p=date_split_new['day_percent'].tolist())
    iT = []
    for i in iGridTest:
        nt = date_split_new.iloc[i]['days_num']
        T = np.random.randint(nt, nt_total - rho, [1])[0]
        iT.append(T)
    return iT

def train(model,
          x,y,c,
          date_split,
          criterion,
          num_epochs,
          device,saveFolder,
          warmup_epochs,
          base_lr,
          batchSize,rho,
          train_sites,val_sites):

    model = model.to(device)
    criterion = criterion.to(device)

    ngrid, nt, nx = x.shape

    # new train and test splitting
    date_split_new = percentage_day(date_split)
    nt_new = date_split_new['days_num'].iloc[train_sites].mean() # 训练集平均时长

    nIterEp = int(np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / len(train_sites) / nt_new)))


    optim = torch.optim.AdamW(model.parameters(),lr=base_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode='min',  factor=0.5, patience=5,verbose=True,min_lr=1e-6)

    scaler = GradScaler(enabled=(device.type == 'cuda'))

    model_name = model.__class__.__name__
    lossFun_name = criterion.__class__.__name__
    if saveFolder is not None:
        if not os.path.isdir(saveFolder):
            os.makedirs(saveFolder)
        runFile = os.path.join(saveFolder, f'run_printLoss.csv')
        rf = open(runFile, 'w+')

    pltRMSE_train = []
    pltRMSE_val = []

    # 早停机制
    early_stop_counter = 0
    early_stop_patience = 10  # 连续 10 个 epoch 无提升就停
    min_delta = 1e-4
    best_val_loss = float('inf')

    print(f"\n--- 开始训练 {model_name} 模型 ({device}) ---")
    for epoch in range(1,num_epochs+1):

        t0 = time.time()

        # ======== Warmup 调整学习率 ========
        if epoch < warmup_epochs:
            warmup_lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optim.param_groups:
                param_group['lr'] = warmup_lr


        model.train()
        total_train_loss = 0
        i_grids = []

        for iIter in range(0, nIterEp):

            iGrid, iT = randomIndex_percentage(train_sites, [batchSize, rho], date_split_new)

            i_grids.append(iGrid)
            xTrain = selectSubset(x, iGrid, iT, rho, c=c)
            yTrain = selectSubset(y, iGrid, iT, rho)

            optim.zero_grad()

            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(xTrain)
                loss = criterion(outputs, yTrain)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            total_train_loss += loss.item()


        avg_train_loss = total_train_loss / nIterEp

        #############################################################################################
        model.eval()
        total_val_loss = 0

        # 验证集迭代次数估算
        nt_val = date_split_new['days_num'].iloc[val_sites].mean()
        nIterEp_test = max(1, int(np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / len(val_sites) / nt_val))))

        with torch.no_grad():
            with autocast(enabled=(device.type == 'cuda')):
                for iIter in range(0, nIterEp_test):
                    iGridTest,iTTest = randomIndex_percentage(val_sites, [batchSize, rho], date_split_new)
                    xTest = selectSubset(x, iGridTest, iTTest, rho, c=c)
                    yTest = selectSubset(y, iGridTest, iTTest, rho)

                    outputs = model(xTest)
                    loss_test = criterion(outputs, yTest)
                    total_val_loss = total_val_loss + loss_test.item()

            avg_val_loss = total_val_loss / nIterEp_test
            # 记录 Loss
            pltRMSE_train.append([epoch, avg_train_loss])
            pltRMSE_val.append([epoch, avg_val_loss])

            if epoch >= warmup_epochs:
                scheduler.step(avg_val_loss)
                if avg_val_loss < best_val_loss - min_delta:
                    best_val_loss = avg_val_loss
                    early_stop_counter = 0

                    # 可选：保存最优模型
                    if saveFolder is not None:
                        modelFile = os.path.join(saveFolder, 'best_model.pt')
                        torch.save(model, modelFile)
                        print(f"    >>> [New Best] Saved model_best.pt (Loss: {best_val_loss:.4f})")
                else:
                    early_stop_counter += 1
                    print(f"EarlyStopping counter: {early_stop_counter}/{early_stop_patience}")

                    if early_stop_counter >= early_stop_patience:
                        print(f"\n 验证集 loss 连续 {early_stop_patience} 个 epoch 未下降，提前停止训练")
                        break
            current_lr = optim.param_groups[0]['lr']
            if current_lr < 1.1e-6 and early_stop_counter >= 3:
                print(f"\nSTOP: 学习率已降至最低 ({current_lr}) 且 Loss 无提升，提前结束。")
                break

        # printing loss
        logStr = ('Epoch {}, time {:.2f}, {}_train {:.3f}, {}_val {:.3f},LR {:.6f}'.format(
            epoch, time.time() - t0, lossFun_name,avg_train_loss,lossFun_name,avg_val_loss,optim.param_groups[0]['lr']))
        logStr_screen = ('Epoch {}, time {:.2f}, {}_train {:.3f}, {}_val {:.3f},LR {:.6f}'.format(
            epoch, time.time() - t0, lossFun_name,avg_train_loss,lossFun_name,avg_val_loss,optim.param_groups[0]['lr']))

        print(logStr_screen)
        # save loss
        if saveFolder is not None:
            rf.write(logStr + '\n')

    if saveFolder is not None:
        rf.close()
        Visualization.visualize_loss(saveFolder,lossFun_name)
    return model


def Interpolation(model,x,y,c,y_mean,y_std,sites_ID,saveFolder,Target_Name,device,window_size):

    model.eval()
    model_name = model.__class__.__name__

    if saveFolder is not None:
        runFile = os.path.join(saveFolder, f'{model_name}_perform.csv')
        rf = open(runFile, 'w')
    # 确保输入是 Numpy 格式以便切片 (x: N, T, F)
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    N_nodes, T_total, n_features = x.shape
    if c is not None:
        c = np.repeat(c[:,np.newaxis,:],T_total,axis=1)
        x = np.concatenate((x, c), axis=2)

    out_dim = model.ny

    print(f"启动高精度插补模式... 总时长: {T_total}, 窗口: {window_size}")
    # --- 2. 核心：滑窗集成预测 (Sliding Window Loop) ---
    # 初始化累加器 (N, T, Out)
    prediction_sum = np.zeros((N_nodes, T_total, out_dim))
    prediction_counts = np.zeros((N_nodes, T_total, out_dim))
    # 计算需要滑动的总步数
    total_steps = T_total - window_size + 1

    if model_name == 'MoE_LSTM':
        gate_global = np.zeros((N_nodes, model.num_experts), dtype=np.float32)

    # 形状为 [站点数, 总滑窗步数, 窗口大小(120)]
    attention_global = np.zeros((N_nodes, total_steps, window_size), dtype=np.float32)

    time_batch_size = 120  # 时间维度的 batch 大小，可根据显存调整
    site_batch_size=10

    with torch.no_grad():
        for site_start in range(0, N_nodes, site_batch_size):
            site_end = min(site_start + site_batch_size, N_nodes)
            current_n_sites = site_end - site_start

            # 截取当前批次站点的全部时序: [current_n_sites, T_total, F]
            x_sites = x[site_start:site_end]
            # --- 内层循环：按时间滑窗批量推理 ---
            for t_idx in range(0, total_steps, time_batch_size):
                t_end_idx = min(t_idx + time_batch_size, total_steps)

                windows = []
                t_starts = []

                for t in range(t_idx, t_end_idx):
                    windows.append(x_sites[:, t:t + window_size, :])
                    t_starts.append(t)

                # windows 堆叠后形状: [B_time, current_n_sites, window_size, F]
                # 展平为模型标准输入: [B_time * current_n_sites, window_size, F]
                x_tensor = torch.tensor(np.array(windows), dtype=torch.float32).to(device)
                B_time = len(t_starts)
                x_tensor = x_tensor.view(B_time * current_n_sites, window_size, -1)

                # 判断模型是否支持输出权重
                if model_name == 'MoE_LSTM':

                    preds, g_weights = model(x_tensor, return_gate_weights=True)
                    # g_weights: [B_time * current_n_sites, num_experts]
                    g_weights = g_weights.view(B_time, current_n_sites, -1)
                    # 静态属性不变，直接取时间批次第0个即可
                    gate_global[site_start:site_end, :] = g_weights[0].cpu().numpy()

                elif hasattr(model, 'attn_block'):
                    preds, attn_weights = model(x_tensor, return_attn=True)
                    # 提取最后一天对过去 120 天的注意力，并重塑回 [B_time, current_n_sites, window_size]
                    attn_weights_np = attn_weights.cpu().numpy()[:, -1, :].reshape(B_time, current_n_sites, window_size)
                else:
                    preds = model(x_tensor)
                    attn_weights_np = None  # 非注意力模型不提取

                # 恢复形状以便累加: [B_time, current_n_sites, window_size, out_dim]
                preds = preds.view(B_time, current_n_sites, window_size, out_dim).cpu().numpy()

                # 将结果累加回全局容器
                for i, t in enumerate(t_starts):
                    prediction_sum[site_start:site_end, t:t + window_size, :] += preds[i]
                    prediction_counts[site_start:site_end, t:t + window_size, :] += 1
                    # 【新增3】：将当前时刻的注意力权重按站点切片存入全局容器
                    if attn_weights_np is not None:
                        attention_global[site_start:site_end, t, :] = attn_weights_np[i]
            print(f"站点进度: {site_end}/{N_nodes} 已完成推理.")
    print("滑动预测完成，正在计算平均值...")
    # 保存门控权重--专家
    if model_name == 'MoE_LSTM' and saveFolder is not None:
        site_names = sites_ID["P_nm"].values if isinstance(sites_ID, pd.DataFrame) else sites_ID
        df_gates = pd.DataFrame(gate_global, index=site_names,
                                columns=[f'Expert_{i}' for i in range(model.num_experts)])
        df_gates.to_csv(os.path.join(saveFolder, f'{model_name}_Gate_Weights.csv'))
        print(f"[*] MoE 专家门控权重已提取并保存！")
    # 保存注意权重--时间上
    if saveFolder is not None and (hasattr(model, 'attn_block')):
        attn_save_path = os.path.join(saveFolder, f'{model_name}_attention_weights.npy')
        np.save(attn_save_path, attention_global)
        print(f"[*] 注意力权重已成功提取并保存至: {attn_save_path}")

    # --- 3. 计算平均值 (Ensemble Result) ---
    # 处理边缘 (计数为0的地方设为1防止除0，虽然step=1通常全覆盖)
    prediction_counts[prediction_counts == 0] = 1
    final_outputs = prediction_sum / prediction_counts  # [N, T, Out]


    site_names = sites_ID["P_nm"].values if isinstance(sites_ID, pd.DataFrame) else sites_ID

    imputed_dfs = {}
    obs_dfs ={}
    for i, var_name in enumerate(Target_Name):
        print(f"\n--- 评估变量: {var_name} ---")

        pred_raw = final_outputs[:, :, i]  # [N, T]
        obs_raw = y[:, :, i]  # [N, T]

        try:
            cur_std = y_std.flat[i] if isinstance(y_std, np.ndarray) else y_std
            cur_mean = y_mean.flat[i] if isinstance(y_mean, np.ndarray) else y_mean
        except:
            cur_std = y_std[:, :, i][0][0]
            cur_mean = y_mean[:, :, i][0][0]

        pred_inv = pred_raw * cur_std + cur_mean
        obs_inv = obs_raw * cur_std + cur_mean

        # 反Log
        pred_final = np.expm1(pred_inv)
        obs_final = np.expm1(obs_inv)

        df_pred = pd.DataFrame(pred_final, index=site_names).T
        df_obs = pd.DataFrame(obs_final, index=site_names).T

        df_obs_clean = df_obs.replace(0, np.nan)

        imputed_dfs[var_name] = df_pred
        obs_dfs[var_name] = df_obs

        if saveFolder:
            filePath = saveFolder + '/out_ep' + f"{model_name}" + f'_{var_name}' + '.csv'
            if os.path.exists(filePath):
                os.remove(filePath)
            df_pred.to_csv(filePath, index=False)


        all_valid_obs = []
        all_valid_preds = []

        #计算指标
        for site in site_names:
            # 只在有真实值的地方计算误差
            mask = (~np.isnan(df_obs_clean[site])) & (~np.isnan(df_pred[site]))

            if np.sum(mask) < 2: continue

            valid_obs = df_obs_clean[site][mask].values
            valid_pred = df_pred[site][mask].values

            all_valid_obs.append(valid_obs)
            all_valid_preds.append(valid_pred)

            r2 = r2_score(valid_obs, valid_pred)
            rmse = np.sqrt(mean_squared_error(valid_obs, valid_pred))
            nse = he.evaluator(he.nse, valid_pred, valid_obs)[0]
            kge, r, alpha, beta = he.kge(valid_pred, valid_obs).squeeze()
            fhv = he.evaluator(he.fhv, valid_pred, valid_obs).squeeze()


            logStr = f'Variable:{var_name}, Site:{site}, R2:{r2:.3f}, NSE:{nse:.3f},KGE:{kge:.3f},FHV{fhv:.3f},RMSE:{rmse:.3f}'
            print(logStr)
            if rf: rf.write(logStr + '\n')

        # --- 计算整体指标 (Overall Performance) ---
        if len(all_valid_obs) > 0:
            # 将所有站点的有效数据拼接到一起
            total_obs = np.concatenate(all_valid_obs)
            total_preds = np.concatenate(all_valid_preds)

            if len(total_obs) > 0:
                # 计算整体指标
                total_r2 = r2_score(total_obs, total_preds)
                total_rmse = np.sqrt(mean_squared_error(total_obs, total_preds))

                total_nse = he.evaluator(he.nse, total_preds, total_obs)[0]
                total_kge, total_r, total_alpha, total_beta = he.kge(total_preds, total_obs).squeeze()
                total_fhv = he.evaluator(he.fhv, total_preds, total_obs).squeeze()

                # 打印并保存
                logStr_overall = f'Variable:{var_name}, == OVERALL ==, R2:{total_r2:.3f}, NSE:{total_nse:.3f},KGE:{total_kge:.3f},FHV:{total_fhv:.3f} RMSE:{total_rmse:.3f}'
                print(logStr_overall)
                if rf: rf.write(logStr_overall + '\n')

    if rf: rf.close()

    return imputed_dfs, obs_dfs



