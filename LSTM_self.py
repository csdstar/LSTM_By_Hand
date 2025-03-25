import torch


def sigmoid(matrix):
    return 1 / (1 + torch.exp(-matrix))


def tanh(matrix):
    exp_pos = torch.exp(matrix)
    exp_neg = torch.exp(-matrix)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)


class LSTM:
    # 暂时先定义一层隐藏层
    def __init__(self, n: int, dimension: int, hidden: list):
        # 输入批次大小
        self.n = n
        # 输入特征维度大小 (input dimension)
        self.d = dimension
        # 隐藏层维度大小 (hidden dimension)
        self.h = hidden

        # 存放每个批次的输入矩阵X n×d
        self.X = torch.zeors(self.n, self.d)
        # 存放隐藏层H n×h
        self.H = torch.zeros(self.n, self.h)
        # 存放输出O n×h
        self.O = torch.zeros(self.n, self.h)

        # 参数矩阵
        self.W_xi = torch.randn(self.d, self.h)  # 输入门
        self.W_xf = torch.randn(self.d, self.h)  # 遗忘门
        self.W_xo = torch.randn(self.d, self.h)  # 输出门

        self.W_hi = torch.randn(self.h, self.h)  # 输入门
        self.W_hf = torch.randn(self.h, self.h)  # 遗忘门
        self.W_ho = torch.randn(self.h, self.h)  # 输出门

        self.B_i = torch.randn(self.n, self.h)
        self.B_f = torch.randn(self.n, self.h)
        self.B_o = torch.randn(self.n, self.h)
