import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import RBF

class AttentionBlock(nn.Module):
    """
    独立的自注意力模块，包含多头注意力、Dropout、残差连接和层归一化。
    """
    def __init__(self, hidden_size, num_heads, drop_rate):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=drop_rate
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        # x: [Batch, Seq_len, hidden_size]
        # Query, Key, Value 均来自输入 x
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.dropout(attn_out)

        # 残差连接与层归一化
        return self.norm(x + attn_out)

class LSTMModel(nn.Module):
    def __init__(self, nx,ny,hidden_size,num_layer, drop_rate,num_heads=4):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size=hidden_size
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(self.nx, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size,num_layers=num_layer, batch_first=True, bidirectional=False)
        self.attn_block = AttentionBlock(self.hidden_size, num_heads, drop_rate)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            self.drop,
            nn.Linear(self.hidden_size, self.ny),
        )
    def forward(self, x):

        B, N, T, _ = x.shape
        x_reshaped = x.reshape(B * N, T, -1)
        x_in = F.relu(self.fc(x_reshaped))
        lstm_out,_ = self.lstm(x_in)
        attn_out = self.attn_block(lstm_out)
        mlp_out = self.mlp(attn_out)

        return mlp_out.reshape(B, N, T, self.ny)


class STGNNModel(nn.Module):
    def __init__(self, nx,ny,num_nodes,edge_index, hidden_size,num_layer, drop_rate,device):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.num_nodes = num_nodes
        self.device = device
        self.hidden_size=hidden_size
        self.drop = nn.Dropout(drop_rate)
        # -------------------------------------------------------
        # 将稀疏的 edge_index 转为稠密矩阵 [N, N]
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].to(device)
        adj = adj + torch.eye(num_nodes).to(device)
        deg = adj.sum(dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        norm_adj = deg_inv.view(-1, 1) * adj
        self.register_buffer('norm_adj', norm_adj)

        # -------------------------------------------------------
        self.fc = nn.Linear(self.nx, self.hidden_size)
        self.lstm1 = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=num_layer,batch_first=True)
        # GCN 公式: A_hat * X * W
        self.W1 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.b1 = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.W2 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.b2 = nn.Parameter(torch.FloatTensor(self.hidden_size))

        # -------------------------------------------------------
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            self.drop,
            nn.Linear(self.hidden_size, self.ny),
        )
        self._init_weights()
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def gcn_layer(self, x,W,b):
        # 公式: Output = Norm_Adj * X * W
        # 利用矩阵乘法: [N, N] x [B*T, N, H] -> [B*T, N, H]
        support = torch.matmul(self.norm_adj, x)
        # 第二步: 线性变换 (Transformation)
        # [B*T, N, H] x [H, H] -> [B*T, N, H]
        gcn_out = torch.matmul(support, W) + b
        return gcn_out

    def forward(self, x):  # x: [N, T, F]

        B, N, T, _ = x.shape
        x_reshaped = x.reshape(B * N, T, -1)
        x_in = self.fc(x_reshaped)
        lstm_out, _ = self.lstm1(x_in)
        # [B*N, T, H] -> [B, N, T, H] -> [B, T, N, H] -> [B*T, N, H]
        x_gcn_in = lstm_out.view(B, N, T, self.hidden_size).permute(0, 2, 1, 3).reshape(B * T, N, self.hidden_size)
        # 3. 执行图卷积 (并行处理 B*T 张图)
        H1 = self.gcn_layer(x_gcn_in,self.W1,self.b1)
        H1 = F.gelu(H1)
        gcn_out= self.gcn_layer(H1,self.W2,self.b2)
        # 还原维度: [B*T, N, H] -> [B, T, N, H] -> [B, N, T, H]
        out = self.mlp(gcn_out)# [B, N, T, Out]
        out = out.view(B, T, N, self.ny).permute(0, 2, 1, 3)

        return out



# ==========================================
# 核心层：物理启发的滞后图卷积 (Physics-Guided GCN)
# ==========================================
class PhysicsGuidedGCN(nn.Module):
    def __init__(self, in_features, hidden_size):
        super(PhysicsGuidedGCN, self).__init__()

        self.W = nn.Linear(in_features, hidden_size)

    def forward(self, x,A_list):
        """
        x: [Batch, N, T, F]
        """
        B, N, T, F_in = x.shape
        max_lag = A_list.shape[1]

        # 调整维度以适应 einsum: [B, T, N, F]
        x_trans = x.permute(0, 2, 1, 3)
        A_list = A_list.to(dtype=x_trans.dtype, device=x_trans.device)
        # 初始化输出容器 [B, T, N, F]
        out_agg = torch.zeros(B, T, N, F_in, device=x.device)

        for lag in range(max_lag):
            norm_A_k = A_list[:,lag,:]

            # ==========================================
            # 核心1：构造多重滞后矩阵，实现上游t-1时刻的水流到下游t时刻
            if lag == 0:
                x_lagged = x_trans
            else:
                x_lagged = torch.roll(x_trans, shifts=lag, dims=1)
                x_lagged[:, :lag, :, :] = 0.0
            # ==========================================
            # 核心2：实现上游节点的水流汇到下游
            agg = torch.einsum('bij,btjf->btif', norm_A_k, x_lagged)
            out_agg += agg
            # ==========================================
        # [B, N, T, F]
        out_final = out_agg.permute(0, 2, 1, 3)
        return F.gelu(self.W(out_final))


class PhysicsSTGNN(nn.Module):
    def __init__(self, nx,ny,  hidden_size,num_layer, drop_rate):
        super(PhysicsSTGNN, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size

        self.fc = nn.Linear(self.nx, self.hidden_size)
        self.gcn = PhysicsGuidedGCN(self.hidden_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size , self.hidden_size , num_layers=num_layer, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size , self.hidden_size ),
            nn.ReLU(),
            nn.Linear(self.hidden_size , self.ny)
        )

    def forward(self, x,A_list):
        # x: [B, N, T, F]
        B, N, T, nF = x.shape
        S_in = F.relu(self.fc(x))

        gnn_out = self.gcn(S_in,A_list)

        T_in = gnn_out.reshape(B*N, T, -1)
        lstm_out, (_,c) = self.lstm(T_in)
        out = self.mlp(lstm_out)

        return out.reshape(B, N, T, -1)



