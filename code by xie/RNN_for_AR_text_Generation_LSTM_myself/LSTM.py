import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=True)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden

        combined = self.i2h(x) + self.h2h(h_prev)
        i_gate, f_gate, c_gate, o_gate = torch.split(combined, self.hidden_size, dim=1)

        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        c_gate = torch.tanh(c_gate)
        o_gate = torch.sigmoid(o_gate)

        c_new = f_gate * c_prev + i_gate * c_gate
        h_new = o_gate * torch.tanh(c_new)

        self.saved_tensors = (x, h_prev, c_prev, i_gate, f_gate, c_gate, o_gate, c_new, h_new)

        return h_new, c_new

    def backward(self, grad_h, grad_c):
        x, h_prev, c_prev, i_gate, f_gate, c_gate, o_gate, c_new, h_new = self.saved_tensors

        # Compute gradients w.r.t. output gate
        grad_o_gate = grad_h * torch.tanh(c_new)
        grad_o_gate *= o_gate * (1 - o_gate)

        # Compute gradients w.r.t. cell state
        grad_c_new = grad_h * o_gate * (1 - torch.tanh(c_new) ** 2) + grad_c
        grad_f_gate = grad_c_new * c_prev
        grad_f_gate *= f_gate * (1 - f_gate)

        grad_i_gate = grad_c_new * c_gate
        grad_i_gate *= i_gate * (1 - i_gate)

        grad_c_gate = grad_c_new * i_gate
        grad_c_gate *= (1 - c_gate ** 2)

        # Compute gradients w.r.t. inputs and weights
        grad_combined = torch.cat([grad_i_gate, grad_f_gate, grad_c_gate, grad_o_gate], dim=1)

        grad_x = self.i2h.weight.t().mm(grad_combined.t()).t()
        grad_h_prev = self.h2h.weight.t().mm(grad_combined.t()).t()
        grad_c_prev = grad_c_new * f_gate

        # Compute weight gradients
        grad_w_i2h = grad_combined.t().mm(x)
        grad_w_h2h = grad_combined.t().mm(h_prev)

        # Return gradients w.r.t. x, h_prev, and c_prev (for backpropagation through time)
        return grad_x, grad_h_prev, grad_c_prev, grad_w_i2h, grad_w_h2h


class MYLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(MYLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [LSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]
        )

    def forward(self, x, hidden=None):
        batch_size, seq_len, input_dim = x.size()

        if hidden is None:
            h_zeros = x.new_zeros((self.num_layers, batch_size, self.hidden_size))
            c_zeros = x.new_zeros((self.num_layers, batch_size, self.hidden_size))
            hidden = (h_zeros, c_zeros)

        h, c = hidden
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_out, c_out = [], []

            for layer in range(self.num_layers):
                h_next, c_next = self.cells[layer](x_t, (h[layer], c[layer]))
                x_t = h_next
                h_out.append(h_next)
                c_out.append(c_next)

            outputs.append(h_next)
            h, c = torch.stack(h_out), torch.stack(c_out)

        return torch.stack(outputs, dim=1), (h, c)

    def backward(self, grad_output, hidden_grad=None):
        if hidden_grad is None:
            h_grad = [torch.zeros_like(grad_output[0])] * self.num_layers
            c_grad = [torch.zeros_like(grad_output[0])] * self.num_layers
        else:
            h_grad, c_grad = hidden_grad

        grad_w_i2h = [torch.zeros_like(self.cells[layer].i2h.weight) for layer in range(self.num_layers)]
        grad_w_h2h = [torch.zeros_like(self.cells[layer].h2h.weight) for layer in range(self.num_layers)]

        for t in reversed(range(len(grad_output))):
            grad_out = grad_output[:, t, :]

            for layer in reversed(range(self.num_layers)):
                cell = self.cells[layer]
                grad_out, h_grad[layer], c_grad[layer], d_w_i2h, d_w_h2h = cell.backward(
                    grad_out + h_grad[layer], c_grad[layer]
                )

                grad_w_i2h[layer] += d_w_i2h
                grad_w_h2h[layer] += d_w_h2h

        return grad_w_i2h, grad_w_h2h
