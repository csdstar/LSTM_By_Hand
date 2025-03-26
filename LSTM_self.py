import torch


def sigmoid(matrix):
    return 1 / (1 + torch.exp(-matrix))


def tanh(matrix):
    exp_pos = torch.exp(matrix)
    exp_neg = torch.exp(-matrix)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)


class LSTM:
    def __init__(self, n: int, dimension: int, h_list: list):
        # 输入批次大小
        self.n = n

        # 初始输入的特征维度大小
        self.d = dimension

        # 存储每个隐藏层的维度大小
        self.h_list = h_list

        # 隐藏层数量
        self.h_num = len(self.h_list)

        # 将初始输入的n×d的d视作上一个隐藏层的输出,便于后续代码实现
        # 此时h_list[0]为d, h_list[m]为第m个隐藏层的h
        self.h_list.insert(0, self.d)

        # X_list[i]存放对于隐藏层i的输入X  n×d -> n×h_m-1
        self.X_list = [torch.zeros(self.n, self.h_list[i]) for i in range(self.h_num)]

        # H_list[i]存放隐藏层i的隐藏输出H  n×h -> n×h_m
        self.H_list = [torch.zeros(self.n, self.h_list[i]) for i in range(1, self.h_num + 1)]

        # C_list[i]存放隐藏层i的记忆元C, 与H形状相同  n×h -> n×h_m
        self.C_list = [torch.zeros(self.n, self.h_list[i]) for i in range(1, self.h_num + 1)]

        # 参数矩阵列表
        # d×h
        self.W_xi_list = [torch.randn(self.h_list[i], self.h_list[i + 1]) for i in range(self.h_num)]  # 输入门
        self.W_xf_list = [torch.randn(self.h_list[i], self.h_list[i + 1]) for i in range(self.h_num)]  # 遗忘门
        self.W_xo_list = [torch.randn(self.h_list[i], self.h_list[i + 1]) for i in range(self.h_num)]  # 输出门
        self.W_xc_list = [torch.randn(self.h_list[i], self.h_list[i + 1]) for i in range(self.h_num)]  # 候选记忆元

        # h×h
        self.W_hi_list = [torch.randn(self.h_list[i + 1], self.h_list[i + 1]) for i in range(self.h_num)]  # 输入门
        self.W_hf_list = [torch.randn(self.h_list[i + 1], self.h_list[i + 1]) for i in range(self.h_num)]  # 遗忘门
        self.W_ho_list = [torch.randn(self.h_list[i + 1], self.h_list[i + 1]) for i in range(self.h_num)]  # 输出门
        self.W_hc_list = [torch.randn(self.h_list[i + 1], self.h_list[i + 1]) for i in range(self.h_num)]  # # 候选记忆元

        # n×h
        self.b_i_list = [torch.randn(n, self.h_list[i + 1]) for i in range(self.h_num)]  # 输入门
        self.b_f_list = [torch.randn(n, self.h_list[i + 1]) for i in range(self.h_num)]  # 遗忘门
        self.b_o_list = [torch.randn(n, self.h_list[i + 1]) for i in range(self.h_num)]  # 输出门
        self.b_c_list = [torch.randn(n, self.h_list[i + 1]) for i in range(self.h_num)]  # 候选记忆元

    # 从序号为index的layer处获取参数
    def get_params_of_layer(self, index: int):
        # index范围是0 ~ h_num - 1
        if index not in range(self.h_num):
            print("Error: get params from unexpected Layer")
            return None

        X = self.X_list[index]
        H = self.H_list[index]
        C = self.C_list[index]

        W_xi = self.W_xi_list[index]
        W_xf = self.W_xf_list[index]
        W_xo = self.W_xo_list[index]
        W_xc = self.W_xc_list[index]

        W_hi = self.W_hi_list[index]
        W_hf = self.W_hf_list[index]
        W_ho = self.W_ho_list[index]
        W_hc = self.W_hc_list[index]

        b_i = self.b_i_list[index]
        b_f = self.b_f_list[index]
        b_o = self.b_o_list[index]
        b_c = self.b_c_list[index]

        return X, H, C, W_xi, W_xf, W_xo, W_xc, W_hi, W_hf, W_ho, W_hc, b_i, b_f, b_o, b_c

    def forward(self):
        h_num = self.h_num
        # 开始逐层前向传播
        for i in range(h_num):
            X, H, C, W_xi, W_xf, W_xo, W_xc, W_hi, W_hf, W_ho, W_hc, b_i, b_f, b_o, b_c = self.get_params_of_layer(i)

            # 遗忘门 n×h
            F = sigmoid(X @ W_xf + H @ W_hf + b_f)

            # 输入门 n×h
            I = sigmoid(X @ W_xi + H @ W_hi + b_i)

            # 输出门 n×h
            O = sigmoid(X @ W_xo + H @ W_ho + b_o)

            # 候选细胞状态 n×h
            C_tilda = tanh(X @ W_xc + H @ W_hc + b_c)

            # 新的细胞状态，由 遗忘门×过去细胞状态 + 输入门×候选细胞状态组成（注意这里是按元素乘法而非矩阵乘法）
            C = F * C + I * C_tilda

            # 新的隐藏状态
            H = O * tanh(C)

            # 存储更新后的记忆元和隐藏元
            self.H_list[i] = H
            self.C_list[i] = C

    def backward(self):
        pass