
import torch
import torch.nn as nn
import torch.nn.functional as F



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
        attn_out, attn_weights = self.attention(x, x, x)
        attn_out = self.dropout(attn_out)

        # 残差连接与层归一化
        return self.norm(x + attn_out),attn_weights



class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_c, out_c, k, padding=(k-1)//2),
            nn.ReLU(),
            nn.MaxPool1d(k,stride=1, padding=(k-1)//2)
        )

    def forward(self, x):
        return self.net(x)



# -------------------------------------------------------------------- #
class LSTMModel(nn.Module):
    def __init__(self, nx,ny,hidden_size,drop_rate):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size=hidden_size
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(self.nx, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size,num_layers=2, batch_first=True, bidirectional=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.ny),
        )
    def forward(self, x):  # x: [N, T, F]
        N, T, _ = x.shape
        x_in = F.relu(self.fc(x))
        lstm_out,_ = self.lstm(x_in)
        mlp_out = self.mlp(lstm_out)
        return mlp_out


class ATTLSTMModel(nn.Module):
    def __init__(self, nx,ny,hidden_size,drop_rate,num_heads=4):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size=hidden_size
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(self.nx, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size,num_layers=2, batch_first=True, bidirectional=False)

        self.attn_block = AttentionBlock(self.hidden_size, num_heads, drop_rate)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            self.drop,
            nn.Linear(self.hidden_size, self.ny),
        )
    def forward(self, x,return_attn=False):  # x: [N, T, F]
        N, T, _ = x.shape
        x_in = F.relu(self.fc(x))
        lstm_out,_ = self.lstm(x_in)
        attn_out,attn_weights = self.attn_block(lstm_out)
        mlp_out = self.mlp(attn_out)
        if return_attn:
            # 推理阶段：返回预测值和权重
            # attn_weights 形状通常为 [Batch, Seq_len, Seq_len]
            return mlp_out, attn_weights
        else:
            return mlp_out



class CNNLSTMmodel(torch.nn.Module):

    def __init__(self, nx, ny,hidden_size,drop_rate,kernel_size=5):

        super(CNNLSTMmodel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        self.features = nn.Sequential()
        self.drop = drop_rate
        self.kernel_size = kernel_size

        self.dense = nn.Linear(self.nx, self.hidden_size)

        input_channel = 2
        # two conv layer
        self.conv = nn.Sequential(
            ConvBlock(input_channel, self.hidden_size, self.kernel_size),
            ConvBlock(self.hidden_size, self.hidden_size, self.kernel_size)
        )


        self.lstm = LSTMModel(self.hidden_size*2, self.hidden_size,self.hidden_size,self.drop)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(self.hidden_size, self.ny),
        )

    def forward(self, x):
        N, T, _ = x.shape
        ## conv layer
        ## 对pre和swe两列特征进行卷积
        z = x[:,:,0:2]
        z = z.permute(0, 2, 1)
        z_conv = self.conv(z)
        z_t = z_conv.permute(0, 2, 1)
        x_fc = self.dense(x)
        x_in = torch.cat([x_fc,z_t],dim=2)

        outLSTM = self.lstm(x_in)
        out = self.mlp(outLSTM)

        return out



class ATCLSTMModel(nn.Module):

    def __init__(self, nx,ny,hidden_size,drop_rate,num_heads=4,kernel_size=5):
        super(ATCLSTMModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        self.features = nn.Sequential()
        self.drop = drop_rate
        self.kernel_size = kernel_size

        self.dense = nn.Linear(self.nx, self.hidden_size)

        input_channel = 2
        # two conv layer
        self.conv = nn.Sequential(
            ConvBlock(input_channel, self.hidden_size, self.kernel_size),
            ConvBlock(self.hidden_size, self.hidden_size, self.kernel_size))

        self.attn_block = AttentionBlock(self.hidden_size, num_heads, drop_rate)

        self.lstm = LSTMModel(self.hidden_size*2, self.hidden_size,self.hidden_size,self.drop)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(self.hidden_size, self.ny))


    def forward(self, x, return_attn=False):
        N, T, _ = x.shape
        ## conv layer
        ## 对pre和swe两列特征进行卷积
        z = x[:,:,0:2]
        z = z.permute(0, 2, 1)
        z_conv = self.conv(z)
        z_t = z_conv.permute(0, 2, 1)
        x_fc = self.dense(x)
        x_in = torch.cat([x_fc,z_t],dim=2)
        outLSTM = self.lstm(x_in)
        attn_out ,attn_weights = self.attn_block(outLSTM)
        mlp_out = self.mlp(attn_out)

        if return_attn:
            return mlp_out, attn_weights
        else:
            return mlp_out

#------------------------------
class MoE_LSTM(nn.Module):
    def __init__(self, nx, ny, nc, hidden_size, drop_rate, num_experts=3):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.nc = nc
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.fc_in = nn.Linear(self.nx, hidden_size)

        # ---------- Experts ----------
        self.experts = nn.ModuleList([
            nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)
            for _ in range(num_experts)
        ])

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, ny)
        )

        self.gate = nn.Sequential(
            nn.Linear(self.nc, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_experts)
        )

    def forward(self, x, return_gate_weights=False):
        B, T, F_dim = x.shape

        x_in = F.relu(self.fc_in(x))

        c = x[:, 0, -self.nc:]  # [B, nc]

        gate_logits = self.gate(c)  # [B, num_experts]
        temperature = 0.3
        gate_weights = F.softmax(gate_logits/temperature, dim=-1)

        expert_outputs = []
        for expert in self.experts:
            out, _ = expert(x_in)  # [B, T, H]
            expert_outputs.append(out)
        # [B, T, H, num_experts]
        expert_stack = torch.stack(expert_outputs, dim=-1)

        # 5. 加权融合
        gate_weights = gate_weights.unsqueeze(1).unsqueeze(1)  # [B,1,1,E]
        out = torch.sum(expert_stack * gate_weights, dim=-1)  # [B,T,H]
        out = self.mlp(out)

        if return_gate_weights:
            return out, gate_weights
        return out

