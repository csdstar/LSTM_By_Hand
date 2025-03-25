import torch
import torch.nn as nn
import torch.optim as optim


# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 全连接层 (输出层)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态 (h0, c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播
        out, _ = self.lstm(x, (h0, c0))  # LSTM输出 (out, (h_n, c_n))

        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


# 示例超参数
input_size = 10  # 输入特征数
hidden_size = 20  # LSTM隐藏层维度
output_size = 1  # 输出维度 (如回归问题中为1)
num_layers = 2  # LSTM层数

# 创建模型
model = LSTMModel(input_size, hidden_size, output_size, num_layers)

# 示例数据
inputs = torch.randn(32, 5, input_size)  # (batch_size, seq_len, input_size)
outputs = model(inputs)
print("模型输出维度:", outputs.shape)  # (32, 1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 示例训练循环
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, torch.randn(32, 1))  # 假设真实值
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')
