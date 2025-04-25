import torch
import torch.nn as nn
import torch.nn.init as init


def sigmoid(matrix):
    return 1 / (1 + torch.exp(-matrix))


def sigmoid_derivative(matrix):
    return sigmoid(matrix) * (1 - sigmoid(matrix))


def tanh(matrix):
    exp_pos = torch.exp(matrix)
    exp_neg = torch.exp(-matrix)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)


def tanh_derivative(matrix):
    return 1 - torch.tanh(matrix) ** 2


# 辅助函数，用于初始化参数
def init_param(shape):
    param = nn.Parameter(torch.empty(*shape))
    init.xavier_uniform_(param)  # xavier参数初始化方式，适用于 tanh/sigmoid
    return param


# 辅助函数：用于累加梯度，自动处理 .grad 初始化为 0 的情况
def accumulate_grad(param, grad):
    if param.grad is None:
        param.grad = torch.zeros_like(param, dtype=param.dtype, device=param.device)

    param.grad += grad.detach().to(param.dtype)


# 辅助函数，用于记录并打印每次梯度迭代过程中产生的最大梯度范数
def check_all_gradients(model):
    # 全局极值
    max_grad_value = -float('inf')
    max_grad_name = None
    max_tensor = None

    min_grad_value = float('inf')
    min_grad_name = None
    min_tensor = None

    def check_and_log(name, tensor, verbose=False):
        if tensor is None or tensor is None:
            print("tensor或grad是none!")
            return
        nonlocal max_grad_value, max_grad_name, max_tensor, min_grad_value, min_grad_name, min_tensor

        # 当前梯度的均值、正则化矩阵、极值
        grad_norm = torch.norm(tensor).item()
        grad_max = tensor.abs().max().item()
        grad_min = tensor.abs().min().item()
        grad_mean = tensor.abs().mean().item()

        if verbose:
            print(f"{name}: grad_norm={grad_norm:.6f}, grad_max={grad_max:.6f}, grad_min={grad_min:.6f}, grad_mean={grad_mean:.6f}")

        if grad_max > max_grad_value:
            max_grad_value = grad_max
            max_grad_name = name
            max_tensor = tensor

        if grad_min < min_grad_value:
            min_grad_value = grad_min
            min_grad_name = name
            min_tensor = tensor

    for i in range(model.h_num):
        check_and_log(f"W_xi[{i}]", model.W_xi_list[i])
        check_and_log(f"W_hi[{i}]", model.W_hi_list[i])
        check_and_log(f"b_i[{i}]", model.b_i_list[i])

        check_and_log(f"W_xf[{i}]", model.W_xf_list[i])
        check_and_log(f"W_hf[{i}]", model.W_hf_list[i])
        check_and_log(f"b_f[{i}]", model.b_f_list[i])

        check_and_log(f"W_xo[{i}]", model.W_xo_list[i])
        check_and_log(f"W_ho[{i}]", model.W_ho_list[i])
        check_and_log(f"b_o[{i}]", model.b_o_list[i])

        check_and_log(f"W_xc[{i}]", model.W_xc_list[i])
        check_and_log(f"W_hc[{i}]", model.W_hc_list[i])
        check_and_log(f"b_c[{i}]", model.b_c_list[i])

        check_and_log(f"W_hq[{i}]", model.W_hq_list[i])
        check_and_log(f"b_hq[{i}]", model.b_hq_list[i])

    print(f"\n⚠️ 当前最大梯度项：{max_grad_name}, grad_max = {max_grad_value:.6f}\n")
    check_and_log(max_grad_name, max_tensor, True)
    print(f"\n⚠️ 当前最小梯度项：{min_grad_name}, grad_min = {min_grad_value:.6f}\n")
    check_and_log(min_grad_name, min_tensor, True)


# 辅助函数，用于检查并替换反向传播中的异常值
def check_tensor(name, tensor, index, fix=True, replace_with=0.0):
    if tensor is None:
        print(f"❌ 第{index}层传播的 {name} 是 None")
        return

    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()

    if has_nan:
        print(f"❌ 第{index}层传播的 {name} 中存在 NaN")
    if has_inf:
        print(f"❌ 第{index}层传播的 {name} 中存在 Inf")

    if fix and (has_nan or has_inf):
        tensor.data = torch.nan_to_num(tensor.data, nan=replace_with, posinf=replace_with, neginf=replace_with)
        print(f"✅ 已将第{index}层的 {name} 中 NaN/Inf 替换为 {replace_with}")


class LSTM(nn.Module):
    def __init__(self, n: int, in_dimension: int, out_dimension, h_list: list):
        super().__init__()
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

        # dropout概率
        self.dropout_rate = 0.3

        # 将初始输入的n×d的d视作上一个隐藏层的输出,便于后续代码实现
        # 此时h_list[0]为d, h_list[m]为第m个隐藏层的h, h_list[m+1]为最终输出的维度大小
        self.h_list.insert(0, self.in_d)
        self.h_list.append(self.out_d)

        # X_list[i]存放对于隐藏层i的输入X  n×d -> n×h_m-1
        self.X_list = [torch.zeros(self.n, self.h_list[i]) for i in range(self.h_num)]

        # Y_list[i]存放对于每个时间步t的输出Yt
        self.Y_list = []

        # H_list[i]存放隐藏层i的隐藏输出H  n×h -> n×h_m
        self.H_list = [torch.zeros(self.n, self.h_list[i]) for i in range(1, self.h_num + 1)]

        # H_prev_list存放隐藏层i上一时间步的记忆元H_t-1
        self.H_prev_list = [torch.zeros(self.n, self.h_list[i]) for i in range(1, self.h_num + 1)]

        # C_list[i]存放隐藏层i当前时间步的记忆元C, 与H形状相同  n×h -> n×h_m
        self.C_list = [torch.zeros(self.n, self.h_list[i]) for i in range(1, self.h_num + 1)]

        # C_prev_list存放隐藏层i上一时间步的记忆元C_t-1
        self.C_prev_list = self.C_list = [torch.zeros(self.n, self.h_list[i]) for i in range(1, self.h_num + 1)]

        # 参数矩阵列表
        # W_x 系列 d × h
        self.W_xi_list = nn.ParameterList([init_param((self.h_list[i], self.h_list[i + 1])) for i in range(self.h_num)])
        self.W_xf_list = nn.ParameterList([init_param((self.h_list[i], self.h_list[i + 1])) for i in range(self.h_num)])
        self.W_xo_list = nn.ParameterList([init_param((self.h_list[i], self.h_list[i + 1])) for i in range(self.h_num)])
        self.W_xc_list = nn.ParameterList([init_param((self.h_list[i], self.h_list[i + 1])) for i in range(self.h_num)])

        # W_h 系列 h × h
        self.W_hi_list = nn.ParameterList([init_param((self.h_list[i + 1], self.h_list[i + 1])) for i in range(self.h_num)])
        self.W_hf_list = nn.ParameterList([init_param((self.h_list[i + 1], self.h_list[i + 1])) for i in range(self.h_num)])
        self.W_ho_list = nn.ParameterList([init_param((self.h_list[i + 1], self.h_list[i + 1])) for i in range(self.h_num)])
        self.W_hc_list = nn.ParameterList([init_param((self.h_list[i + 1], self.h_list[i + 1])) for i in range(self.h_num)])

        # 偏置项 1×h，在加法时会自动被拓展为n×h
        self.b_i_list = nn.ParameterList([nn.Parameter(torch.zeros(1, self.h_list[i + 1])) for i in range(self.h_num)])  # 输入门
        self.b_f_list = nn.ParameterList([nn.Parameter(torch.zeros(1, self.h_list[i + 1])) for i in range(self.h_num)])  # 遗忘门
        self.b_o_list = nn.ParameterList([nn.Parameter(torch.zeros(1, self.h_list[i + 1])) for i in range(self.h_num)])  # 输出门
        self.b_c_list = nn.ParameterList([nn.Parameter(torch.zeros(1, self.h_list[i + 1])) for i in range(self.h_num)])  # 候选记忆元

        # n×h, Z = XW + HW + b
        self.Z_i_list = [torch.zeros(n, self.h_list[i + 1]) for i in range(self.h_num)]
        self.Z_f_list = [torch.zeros(n, self.h_list[i + 1]) for i in range(self.h_num)]
        self.Z_o_list = [torch.zeros(n, self.h_list[i + 1]) for i in range(self.h_num)]
        self.Z_c_list = [torch.zeros(n, self.h_list[i + 1]) for i in range(self.h_num)]

        # n×h
        self.I_list = [torch.zeros(n, self.h_list[i + 1]) for i in range(self.h_num)]
        self.F_list = [torch.ones(n, self.h_list[i + 1]) for i in range(self.h_num)]
        self.O_list = [torch.zeros(n, self.h_list[i + 1]) for i in range(self.h_num)]
        self.C_tilda_list = [torch.zeros(n, self.h_list[i + 1]) for i in range(self.h_num)]

        # h×h，最后一项改为h×out_d
        self.W_hq_list = nn.ParameterList(
            [nn.Parameter(torch.randn(self.h_list[i + 1], self.h_list[i + 1]) if i < self.h_num - 1 else torch.randn(self.h_list[i + 1], self.h_list[i + 2])) for i in range(self.h_num)])

        # 1×h_m+1(out_d)
        self.b_hq_list = nn.ParameterList([nn.Parameter(torch.randn(1, self.h_list[i + 1]) if i < self.h_num - 1 else torch.randn(1, self.h_list[i + 2])) for i in range(self.h_num)])

        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-5)

    # 从序号为index的layer处获取参数
    def get_backward_params_of_layer(self, index: int):
        # index范围是0 ~ h_num - 1
        if index not in range(self.h_num):
            print("Error: get params from unexpected Layer")
            return None

        X = self.X_list[index].detach()
        H = self.H_list[index].detach()
        H_prev = self.H_prev_list[index].detach()

        W_xi = self.W_xi_list[index].detach()
        W_xf = self.W_xf_list[index].detach()
        W_xo = self.W_xo_list[index].detach()
        W_xc = self.W_xc_list[index].detach()

        W_hi = self.W_hi_list[index].detach()
        W_hf = self.W_hf_list[index].detach()
        W_ho = self.W_ho_list[index].detach()
        W_hc = self.W_hc_list[index].detach()

        b_i = self.b_i_list[index].detach()
        b_f = self.b_f_list[index].detach()
        b_o = self.b_o_list[index].detach()
        b_c = self.b_c_list[index].detach()

        Z_i = self.Z_i_list[index].detach()
        Z_f = self.Z_f_list[index].detach()
        Z_o = self.Z_o_list[index].detach()
        Z_c = self.Z_c_list[index].detach()

        I = self.I_list[index].detach()
        F = self.F_list[index].detach()
        O = self.O_list[index].detach()
        C = self.C_list[index].detach()

        C_prev = self.C_prev_list[index].detach()
        C_tilda = self.C_tilda_list[index].detach()

        W_q = self.W_hq_list[index].detach()
        b_q = self.b_hq_list[index].detach()

        return X, H, H_prev, W_xi, W_xf, W_xo, W_xc, W_hi, W_hf, W_ho, W_hc, b_i, b_f, b_o, b_c, Z_i, Z_f, Z_o, Z_c, I, F, O, C, C_prev, C_tilda, W_q, b_q

    def get_forward_params_of_layer(self, index: int):
        # index范围是0 ~ h_num - 1
        if index not in range(self.h_num):
            print("Error: get params from unexpected Layer")
            return None

        # 克隆输入、状态、权重和偏置，避免意外修改
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

    # 前向传播,输入X的形状为(seq_len, batch, input_dimension)
    def forward(self, input):
        # 初始化参数
        h_num = self.h_num
        seq_len = input.shape[0]
        Y = None
        self.Y_list = []

        # 逐时间步从最底层输入
        for t in range(seq_len):
            self.X_list[0] = input[t]

            # 开始逐层前向传播
            for i in range(h_num):
                # print(f"前向传播至第{i}层")
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
                self.Z_o_list[i] = Z_o

                O = sigmoid(Z_o)
                self.O_list[i] = O

                # 候选细胞状态C_tilda n×h
                C_tilda = tanh(X @ W_xc + H @ W_hc + b_c)
                self.C_tilda_list[i] = C_tilda

                # 存储旧的记忆元状态C_t-1，便于计算反向传播
                self.C_prev_list[i] = C

                # 新的细胞状态，由 遗忘门×过去细胞状态 + 输入门×候选细胞状态组成（注意这里是按元素乘法而非矩阵乘法）
                C = F * C + I * C_tilda

                # 存储旧的隐藏层状态H_t-1 n×h
                self.H_prev_list[i] = H

                # 新的隐藏状态 n×h
                H = O * tanh(C)

                # 存储更新后的记忆元和隐藏元
                self.H_list[i] = H
                self.C_list[i] = C

                # 计算当前层的输出
                Y = H @ W_q + b_q

                # 上一层的输出作为下一层的输入
                if i < h_num - 1:
                    self.X_list[i + 1] = Y

            # 最后一层的输出作为当前时间步的最终输出
            self.Y_list.append(Y)

        # 返回最后一时间步的最终输出Y, n×out_dim
        return Y

    # 反向传播, 输入每个时间步的梯度dY
    def backward(self, dY_list):
        # print("\n开始反向传播\n")
        _dY_list = dY_list

        # 初始化清空梯度
        self.optimizer.zero_grad()

        # 初始化每层的由下一时间步传上来的H和C梯度列表
        dH_next_list = [None for _ in range(self.h_num)]
        dC_next_list = [None for _ in range(self.h_num)]

        # 逐时间步反向传播
        for t in reversed(range(len(_dY_list))):
            # print(f"反向传播至时间步{t}")
            # 初始化每个时间步传输最顶层的dY, dH, dC
            dY = _dY_list[t].clone()

            for i in reversed(range(self.h_num)):
                # 获取上一时间步传来的dH和dC
                dH_next = dH_next_list[i]
                dC_next = dC_next_list[i]

                # 逐层反向传播
                dY, dH_next, dC_next = self.layer_backward(i, dY, dH_next, dC_next)

                # 更新梯度传递列表
                dH_next_list[i] = dH_next
                dC_next_list[i] = dC_next

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)

        self.optimizer.step()

    # 单层的反向传播
    def layer_backward(self, index, dY, dH_next=None, dC_next=None):
        X, H, H_prev, W_xi, W_xf, W_xo, W_xc, W_hi, W_hf, W_ho, W_hc, b_i, b_f, b_o, b_c, Z_i, Z_f, Z_o, Z_c, I, F, O, C, C_prev, C_tilda, W_q, b_q = self.get_backward_params_of_layer(index)

        # 对部分非可更新梯度单独裁剪，防止跨层梯度爆炸
        max_norm = 5.0

        # 检查基本变量
        tensors_to_check = [
            ("dY", dY), ("H", H), ("H_prev", H_prev),
            ("C", C), ("C_prev", C_prev),
            ("Z_i", Z_i), ("Z_f", Z_f), ("Z_o", Z_o), ("Z_c", Z_c),
            ("I", I), ("F", F), ("O", O), ("C_tilda", C_tilda),
        ]
        for name, tensor in tensors_to_check:
            check_tensor(name, tensor, index)

        if dH_next is None:
            _dH_next = torch.zeros_like(H)
        else:
            _dH_next = dH_next.detach()
        if dC_next is None:
            _dC_next = torch.zeros_like(C)
        else:
            _dC_next = dC_next.detach()

        # 前半部分是来自当前时间步Y_t的梯度
        # 由于每个Y计算时都使用了Ht-1，故要再算上Y_t+1计算出的H的梯度dH_next
        dH = dY @ W_q.t() + _dH_next

        dW_q = H.t() @ dY
        db_q = dY.sum(dim=0, keepdim=True)
        dO = dH * tanh(C)

        # 同理,C的梯度是由H传递下来的
        dC = O * dH * tanh_derivative(C) + _dC_next

        dF = dC * C_prev
        dI = dC * C_tilda
        dC_prev = dC * F
        dC_tilda = dC * I

        dZ_i = sigmoid_derivative(Z_i) * dI
        dZ_f = sigmoid_derivative(Z_f) * dF
        dZ_o = sigmoid_derivative(Z_o) * dO
        dZ_c = tanh_derivative(Z_c) * dC_tilda

        dW_xi = X.t() @ dZ_i
        dW_hi = H_prev.t() @ dZ_i
        db_i = dZ_i.sum(dim=0, keepdim=True)  # b 是 n×h 的，要按 batch 做 sum，下面同理

        dW_xf = X.t() @ dZ_f
        dW_hf = H_prev.t() @ dZ_f
        db_f = dZ_f.sum(dim=0, keepdim=True)

        dW_xo = X.t() @ dZ_o
        dW_ho = H_prev.t() @ dZ_o
        db_o = dZ_o.sum(dim=0, keepdim=True)

        dW_xc = X.t() @ dZ_c
        dW_hc = H_prev.t() @ dZ_c
        db_c = dZ_c.sum(dim=0, keepdim=True)

        # X的梯度是四个门方向的梯度之和
        dX = dZ_i @ W_xi.t() + dZ_f @ W_xf.t() + dZ_o @ W_xo.t() + dZ_c @ W_xc.t()
        norm = dX.norm()
        if norm > max_norm:
            dX = dX * (max_norm / (norm + 1e-6))  # 加 eps 防止除 0

        # 计算传递到前一时刻隐藏状态的梯度
        # 这部分梯度来源于所有门中，Ht-1 参与了各自的线性运算（作为递归部分）
        dH_prev = dZ_i @ W_hi.t() + dZ_f @ W_hf.t() + dZ_o @ W_ho.t() + dZ_c @ W_hc.t()

        # 累加每个时间步的梯度
        accumulate_grad(self.W_xi_list[index], dW_xi)
        accumulate_grad(self.W_hi_list[index], dW_hi)
        accumulate_grad(self.b_i_list[index], db_i)

        accumulate_grad(self.W_xf_list[index], dW_xf)
        accumulate_grad(self.W_hf_list[index], dW_hf)
        accumulate_grad(self.b_f_list[index], db_f)

        accumulate_grad(self.W_xo_list[index], dW_xo)
        accumulate_grad(self.W_ho_list[index], dW_ho)
        accumulate_grad(self.b_o_list[index], db_o)

        accumulate_grad(self.W_xc_list[index], dW_xc)
        accumulate_grad(self.W_hc_list[index], dW_hc)
        accumulate_grad(self.b_c_list[index], db_c)

        accumulate_grad(self.W_hq_list[index], dW_q)
        accumulate_grad(self.b_hq_list[index], db_q)

        return dX, dH_prev, dC_prev

    def clear_memory(self):
        """
        清理内存，避免内存泄漏
        """
        # 清除 X_list, Y_list, H_list 和 C_list 的张量内容
        self.X_list = [torch.zeros(self.n, self.h_list[i]) for i in range(self.h_num)]
        self.Y_list = []  # 清空每次时间步的输出
        self.H_list = [torch.zeros(self.n, self.h_list[i]) for i in range(1, self.h_num + 1)]
        self.C_list = [torch.zeros(self.n, self.h_list[i]) for i in range(1, self.h_num + 1)]
        self.C_prev_list = self.C_list  # 重新赋值避免重复内存

        # 清除存储在计算图中的中间变量（Z, I, F, O, C_tilda）
        self.Z_i_list = [torch.zeros(self.n, self.h_list[i + 1]) for i in range(self.h_num)]
        self.Z_f_list = [torch.zeros(self.n, self.h_list[i + 1]) for i in range(self.h_num)]
        self.Z_o_list = [torch.zeros(self.n, self.h_list[i + 1]) for i in range(self.h_num)]
        self.Z_c_list = [torch.zeros(self.n, self.h_list[i + 1]) for i in range(self.h_num)]

        self.I_list = [torch.zeros(self.n, self.h_list[i + 1]) for i in range(self.h_num)]
        self.F_list = [torch.ones(self.n, self.h_list[i + 1]) for i in range(self.h_num)]
        self.O_list = [torch.zeros(self.n, self.h_list[i + 1]) for i in range(self.h_num)]
        self.C_tilda_list = [torch.zeros(self.n, self.h_list[i + 1]) for i in range(self.h_num)]
