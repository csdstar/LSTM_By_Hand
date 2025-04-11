import numpy as np
import torch
import torch.nn as nn


def sigmoid(matrix):
    return 1 / (1 + torch.exp(-matrix))


def sigmoid_derivative(matrix):
    return sigmoid(matrix) * (1 - sigmoid(matrix))


def tanh(matrix):
    exp_pos = torch.exp(matrix)
    exp_neg = torch.exp(-matrix)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)


def tanh_derivative(matrix):
    return 1 - np.tanh(matrix) ** 2


class LSTM:
    def __init__(self, n: int, in_dimension: int, out_dimension, h_list: list):
        # 输入批次大小
        self.n = n

        # 初始输入的特征维度大小
        self.in_d = in_dimension

        # 最终输出的特征维度大小
        self.out_d = out_dimension

        # 存储每个隐藏层的维度大小
        self.h_list = h_list

        # 隐藏层数量
        self.h_num = len(self.h_list)

        # 将初始输入的n×d的d视作上一个隐藏层的输出,便于后续代码实现
        # 此时h_list[0]为d, h_list[m]为第m个隐藏层的h, h_list[m+1]为最终输出的维度大小
        self.h_list.insert(0, self.in_d)
        self.h_list.append(out_dimension)

        # X_list[i]存放对于隐藏层i的输入X  n×d -> n×h_m-1
        self.X_list = [torch.zeros(self.n, self.h_list[i]) for i in range(self.h_num)]

        # H_list[i]存放隐藏层i的隐藏输出H  n×h -> n×h_m
        self.H_list = [torch.zeros(self.n, self.h_list[i]) for i in range(1, self.h_num + 1)]

        # C_list[i]存放隐藏层i当前时间步的记忆元C, 与H形状相同  n×h -> n×h_m
        self.C_list = [torch.zeros(self.n, self.h_list[i]) for i in range(1, self.h_num + 1)]

        # C_old_list存放隐藏层i上一时间步的记忆元C_t-1
        self.C_old_list = self.C_list = [torch.zeros(self.n, self.h_list[i]) for i in range(1, self.h_num + 1)]

        # 参数矩阵列表
        # d×h
        self.W_xi_list = nn.ParameterList([nn.Parameter(torch.randn(self.h_list[i], self.h_list[i + 1])) for i in range(self.h_num)])  # 输入门
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

        # n×h, Z = XW + HW + b
        self.Z_i_list = [torch.randn(n, self.h_list[i + 1]) for i in range(self.h_num)]
        self.Z_f_list = [torch.randn(n, self.h_list[i + 1]) for i in range(self.h_num)]
        self.Z_o_list = [torch.randn(n, self.h_list[i + 1]) for i in range(self.h_num)]
        self.Z_c_list = [torch.randn(n, self.h_list[i + 1]) for i in range(self.h_num)]

        # 每层隐藏层的最终输出
        # h_m×h_m+1(out_d)
        self.W_hq_list = [torch.randn(self.h_list[i + 1], self.h_list[i + 2]) for i in range(self.h_num)]

        # n×h_m+1(out_d)
        self.b_hq_list = [(n, self.h_list[i + 2]) for i in range(self.h_num)]

        self.I_list = []
        self.F_list = []
        self.O_list = []
        self.C_tilda_list = []

    # 从序号为index的layer处获取参数
    def get_backward_params_of_layer(self, index: int):
        # index范围是0 ~ h_num - 1
        if index not in range(self.h_num):
            print("Error: get params from unexpected Layer")
            return None

        X = self.X_list[index]
        H = self.H_list[index]

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

        Z_i = self.Z_i_list[index]
        Z_f = self.Z_f_list[index]
        Z_o = self.Z_o_list[index]
        Z_c = self.Z_c_list[index]

        I = self.I_list[index]
        F = self.F_list[index]
        O = self.O_list[index]
        C = self.C_list[index]

        C_old = self.C_old_list[index]
        C_tilda = self.C_tilda_list[index]

        W_q = self.W_hq_list[index]
        b_q = self.b_hq_list[index]

        return X, H, W_xi, W_xf, W_xo, W_xc, W_hi, W_hf, W_ho, W_hc, b_i, b_f, b_o, b_c, Z_i, Z_f, Z_o, Z_c, I, F, O, C, C_old, C_tilda, W_q, b_q

    def get_forward_params_of_layer(self, index: int):
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

        W_q = self.W_hq_list[index]
        b_q = self.b_hq_list[index]

        return X, H, C, W_xi, W_xf, W_xo, W_xc, W_hi, W_hf, W_ho, W_hc, b_i, b_f, b_o, b_c, W_q, b_q

    # 前向传播
    def forward(self):
        h_num = self.h_num
        # 开始逐层前向传播
        for i in range(h_num):
            X, H, C, W_xi, W_xf, W_xo, W_xc, W_hi, W_hf, W_ho, W_hc, b_i, b_f, b_o, b_c, W_q, b_q = self.get_forward_params_of_layer(i)

            # 输入门I n×h
            Z_i = X @ W_xi + H @ W_hi + b_i
            self.Z_i_list[i] = Z_i
            I = sigmoid(Z_i)
            self.I_list[i] = I

            # 遗忘门F n×h
            Z_f = X @ W_xf + H @ W_hf + b_f
            self.Z_f_list[i] = Z_f
            F = sigmoid(Z_f)
            self.F_list[i] = F

            # 输出门0 n×h
            Z_o = X @ W_xo + H @ W_ho + b_o
            self.Z_o_list = Z_o
            O = sigmoid(Z_o)
            self.O_list[i] = O

            # 候选细胞状态C_tilda n×h
            C_tilda = tanh(X @ W_xc + H @ W_hc + b_c)
            self.C_tilda_list[i] = C_tilda

            # 记录C_t-1，便于计算反向传播
            self.C_old_list[i] = C

            # 新的细胞状态，由 遗忘门×过去细胞状态 + 输入门×候选细胞状态组成（注意这里是按元素乘法而非矩阵乘法）
            C = F * C + I * C_tilda

            # 新的隐藏状态 n×h
            H = O * tanh(C)

            # 存储更新后的记忆元和隐藏元
            self.H_list[i] = H
            self.C_list[i] = C

            # 计算最终输出
            Y = H @ W_q + b_q

            # 上一层的输出作为下一层的输入
            if i < h_num:
                self.X_list[i + 1] = Y
            else:
                return Y

    def backward(self):
        # 初始化存放每层的梯度
        dW_xi_list = [torch.zeros_like(W) for W in self.W_xi_list]
        dW_hi_list = [torch.zeros_like(W) for W in self.W_hi_list]
        dW_xf_list = [torch.zeros_like(W) for W in self.W_xf_list]
        dW_hf_list = [torch.zeros_like(W) for W in self.W_hf_list]
        dW_xo_list = [torch.zeros_like(W) for W in self.W_xo_list]
        dW_ho_list = [torch.zeros_like(W) for W in self.W_ho_list]
        dW_xc_list = [torch.zeros_like(W) for W in self.W_xc_list]
        dW_hc_list = [torch.zeros_like(W) for W in self.W_hc_list]
        dW_q_list = [torch.zeros_like(W) for W in self.W_hq_list]
        db_q_list = [torch.zeros((1, W.shape[1])) for W in self.W_hq_list]  # n × d_out

    # 单层的反向传播
    def layer_backward(self, index, dY):
        X, H, W_xi, W_xf, W_xo, W_xc, W_hi, W_hf, W_ho, W_hc, b_i, b_f, b_o, b_c, Z_i, Z_f, Z_o, Z_c, I, F, O, C, C_old, C_tilda, W_q, b_q = self.get_backward_params_of_layer(index)
        dH = dY @ W_q.t()
        dW_q = H.t() @ dY
        db_q = dY
        dO = dH * tanh(C)
        dC = O * dH * tanh_derivative(C)

        dF = dC * C_old
        dI = dC * C_tilda
        dC_tilda = dC * I

        db_i = tanh_derivative(Z_i) * dI
        dW_xi = X.t() @ db_i
        dW_hi = H.t() @ db_i

        db_f = tanh_derivative(Z_f) * dF
        dW_xf = X.t() @ db_f
        dW_hf = H.t() @ db_f

        db_o = tanh_derivative(Z_c) * dO
        dW_xo = X.t() @ db_o
        dW_ho = H.t() @ db_o

        db_c = tanh_derivative(Z_c) * dC_tilda
        dW_xc = X.t() @ db_c
        dW_hc = H.t() @ db_c

        dX = db_i @ W_xi.t()
        # if dX != db_c @ W_xc.t():
        #     print("Error")
