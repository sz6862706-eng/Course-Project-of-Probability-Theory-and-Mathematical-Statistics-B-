import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class StockDataset(Dataset):
    def __init__(self, data, seq_len=20):
        self.seq_len = seq_len
        self.data = data

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, rnn_type="RNN"):
        super().__init__()
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == "RNN":
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)


class SeriesRNNModel(nn.Module):
    """串联RNN模型 - 不同类型的RNN层依次串联"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        # 第一层：LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 第二层：GRU（输入是LSTM的输出）
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # 第三层：普通RNN（输入是GRU的输出）
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # LSTM层
        out, _ = self.lstm(x)
        
        # GRU层（接收LSTM的输出）
        out, _ = self.gru(out)
        
        # RNN层（接收GRU的输出）
        out, _ = self.rnn(out)
        
        # 取最后一个时间步
        out = out[:, -1, :]
        
        return self.fc(out)


class ParallelRNNModel(nn.Module):
    """并联RNN模型 - 多个RNN并行处理，然后融合"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, fusion_method="concat"):
        super().__init__()
        self.fusion_method = fusion_method
        
        # 三个并行的RNN分支
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # 根据融合方法确定全连接层的输入维度
        if fusion_method == "concat":
            # 拼接：3个hidden_size
            fc_input_size = hidden_size * 3
        elif fusion_method == "sum" or fusion_method == "mean":
            # 求和或平均：保持hidden_size
            fc_input_size = hidden_size
        elif fusion_method == "attention":
            # 注意力机制：hidden_size
            fc_input_size = hidden_size
            # 注意力权重层
            self.attention = nn.Linear(hidden_size, 1)
        
        self.fc = nn.Linear(fc_input_size, 1)
        
    def forward(self, x):
        # 三个RNN分支并行处理
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(x)
        rnn_out, _ = self.rnn(x)
        
        # 取每个分支的最后一个时间步
        lstm_last = lstm_out[:, -1, :]  # (batch, hidden_size)
        gru_last = gru_out[:, -1, :]    # (batch, hidden_size)
        rnn_last = rnn_out[:, -1, :]    # (batch, hidden_size)
        
        # 融合三个分支的输出
        if self.fusion_method == "concat":
            # 方法1：拼接
            fused = torch.cat([lstm_last, gru_last, rnn_last], dim=1)
            
        elif self.fusion_method == "sum":
            # 方法2：求和
            fused = lstm_last + gru_last + rnn_last
            
        elif self.fusion_method == "mean":
            # 方法3：平均
            fused = (lstm_last + gru_last + rnn_last) / 3
            
        elif self.fusion_method == "attention":
            # 方法4：注意力机制加权融合
            # 堆叠三个输出
            stacked = torch.stack([lstm_last, gru_last, rnn_last], dim=1)  # (batch, 3, hidden_size)
            
            # 计算注意力权重
            attention_scores = self.attention(stacked)  # (batch, 3, 1)
            attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, 3, 1)
            
            # 加权求和
            fused = torch.sum(stacked * attention_weights, dim=1)  # (batch, hidden_size)
        
        return self.fc(fused)
