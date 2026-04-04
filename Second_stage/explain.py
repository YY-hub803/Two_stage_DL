import os
import pandas as pd
import torch
import torch.nn.functional as F


def quantify_pct(model, x, A_list, target_node_idx, target_time_idx,ID,date,save_folder,steps=50):
    """
        使用 Integrated Gradients (IG) 反推来水比例

        参数:
            x: 真实输入产流 [1, N, T, F_in] (确保 Batch=1)
            steps: 积分步数，通常 50 到 100 即可获得高精度
        """
    model.eval()
    x = torch.from_numpy(x).unsqueeze(0).float().to('cuda')
    A_list = A_list.unsqueeze(0)
    # 1. 定义物理基准线：全流域产流为 0
    baseline = torch.zeros_like(x)

    # 2. 存储沿路径累加的梯度
    integrated_gradients = torch.zeros_like(x[0, :, :, 0])

    # 3. 沿直线路径进行积分插值
    with torch.backends.cudnn.flags(enabled=False):
        for alpha in torch.linspace(0.0, 1.0, steps=steps):
            # 构造路径上的插值输入
            interpolated_x = baseline + alpha * (x - baseline)
            interpolated_x.requires_grad_(True)

            # 前向传播与反向传播
            out = model(interpolated_x, A_list)
            target_pred = out[0, target_node_idx, target_time_idx, 0]

            model.zero_grad()
            target_pred.backward()

            # 累加梯度 (取第0个特征: 流量)
            integrated_gradients += interpolated_x.grad[0, :, :, 0]

    # 4. 计算平均梯度
    avg_gradients = integrated_gradients / steps

    # 5. IG 公式: (Input - Baseline) * Average_Gradient
    # 因为 baseline 是 0，所以就是 x * avg_gradients
    actual_inputs = x[0, :, :, 0]
    attributions = actual_inputs * avg_gradients

    # 6. 计算来水比例
    # 保留正向贡献（物理意义上的水流汇集）
    positive_attributions = F.relu(attributions)
    total_attribution = positive_attributions.sum() + 1e-8

    proportion_matrix = positive_attributions / total_attribution
    numpy_data = proportion_matrix.detach().cpu().numpy()

    df_pct_matrix = pd.DataFrame(
        numpy_data,
        index=ID,
        columns=date,
    )
    target_ID_nm = ID[target_node_idx]
    target_time = date[target_time_idx]
    df_pct_matrix.to_csv(os.path.join(save_folder, f"case_{target_ID_nm}_{target_time}.csv"))
    print("事例解释已保存")
    return df_pct_matrix

def quantify_global_pct(model, x, A_list, target_node_idx,ID,save_folder,steps=50):
    """
        使用 Integrated Gradients (IG) 反推来水比例
        参数:
            x: 真实输入产流 [1, N, T, F_in] (确保 Batch=1)
            steps: 积分步数，通常 50 到 100 即可获得高精度
        """
    model.eval()
    x = torch.from_numpy(x).unsqueeze(0).float().to('cuda')
    A_list = A_list.unsqueeze(0)
    # 1. 定义物理基准线：全流域产流为 0
    baseline = torch.zeros_like(x)

    # 2. 存储沿路径累加的梯度
    integrated_gradients = torch.zeros_like(x[:, :, :, 0])

    # 3. 沿直线路径进行积分插值
    with torch.backends.cudnn.flags(enabled=False):
        for alpha in torch.linspace(0.0, 1.0, steps=steps):
            # 构造路径上的插值输入
            interpolated_x = baseline + alpha * (x - baseline)
            interpolated_x.requires_grad_(True)

            # 前向传播与反向传播
            out = model(interpolated_x, A_list)
            target_pred = out[:, target_node_idx, :, 0].sum()

            model.zero_grad()
            target_pred.backward()

            # 累加梯度 (取第0个特征: 流量)
            integrated_gradients += interpolated_x.grad[:, :, :, 0]

    # 4. 计算平均梯度
    avg_gradients = integrated_gradients / steps

    # 5. IG 公式: (Input - Baseline) * Average_Gradient
    # 因为 baseline 是 0，所以就是 x * avg_gradients
    actual_inputs = x[:, :, :, 0]
    attributions = actual_inputs * avg_gradients

    # 6. 计算来水比例
    # 保留正向贡献（物理意义上的水流汇集）
    positive_attributions = F.relu(attributions)
    # 【全局聚合】：沿着 Batch (dim=0) 和 时间 (dim=2) 维度求和
    # 得到每个节点在整个周期内的总绝对贡献量，形状变为 [N]
    global_node_contribution = positive_attributions.sum(dim=(0, 2))

    total_attribution = positive_attributions.sum() + 1e-8
    global_proportion = global_node_contribution / total_attribution

    numpy_data = global_proportion.detach().cpu().numpy()

    df_global_pct = pd.DataFrame(
        numpy_data,
        index=ID,
    )

    df_global_pct.to_csv(os.path.join(save_folder, "global_pct.csv"))
    print("全局解释已保存")
    return df_global_pct

