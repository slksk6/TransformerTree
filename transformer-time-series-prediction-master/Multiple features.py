import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# 从 Excel 文件读取数据
file_path = 'your_file_path.xlsx'  # 替换成你的文件路径
df = pd.read_excel(file_path)

# 假设 Excel 文件中有特征数据 X、时间数据 time 和目标值数据 y
X = df[['feature1', 'feature2', 'feature3']]  # 假设有3个特征
time = df['time']  # 时间数据
y = df['target']  # 目标值数据

# 对特征数据进行归一化处理
X = (X - X.mean()) / X.std()


# 构建时序数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, time, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.time = pd.to_datetime(time)  # 将时间数据转换为日期时间类型
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.time[idx], self.y[idx]


# 创建数据加载器
dataset = TimeSeriesDataset(X, time, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        self.d_model = input_dim
        self.pos_encoder = PositionalEncoding(input_dim, 0.1)
        self.transformer = nn.Transformer(d_model=input_dim, nhead=num_heads, num_encoder_layers=num_layers)
        self.decoder = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer(src, src)
        output = self.decoder(output[-1, :])  # 假设只取最后一个时间步的输出作为预测值
        return output

# 初始化模型
input_dim = 3  # 假设有3个特征
output_dim = 1  # 假设只有一个目标值
num_heads = 2
num_layers = 2
model = TransformerModel(input_dim, output_dim, num_heads, num_layers)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    for batch_X, batch_time, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
