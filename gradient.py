import numpy as np


def output_layer_backward(dY, H, W_q):
    """
    计算从输出层反向传播得到的 dH
    原公式为 Y = H · W_q + b_q

    参数：
    - dY: ∂L/∂Y，形状为 (n, d_out)
    - H: LSTM 最后输出的隐藏状态，形状为 (n, h)
    - W_q: 输出权重矩阵，形状为 (h, d_out)

    返回：
    - dH: ∂L/∂H，形状为 (n, h)
    - dW_q: ∂L/∂W_q，形状为 (h, d_out)
    - db_q: ∂L/∂b_q，形状为 (1, d_out)
    """
    dH = dY @ W_q.T
    dW_q = H.T @ dY
    db_q = np.sum(dY, axis=0, keepdims=True)
    return dH, dW_q, db_q


def layer_backward(dY, H, W_q):
    dH = dY @ W_q.T
    dW_q = H.T @ dY
    db_q = dY.sum(dim=0, keepdim=True)
    return dH, dW_q, db_q

