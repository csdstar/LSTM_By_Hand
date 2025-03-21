import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)

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
        
        return h_new, c_new


class MYLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = nn.ModuleList([LSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        
        if hidden is None:
            h_zeros = x.new_zeros((self.num_layers, batch_size, self.hidden_size))
            c_zeros = x.new_zeros((self.num_layers, batch_size, self.hidden_size))
            hidden = (h_zeros, c_zeros)
        
        h, c = hidden
        layer_out = x
        
        for t in range(seq_len):
            x_t = layer_out[:, t, :]
            h_out, c_out = [], []
            
            for layer in range(self.num_layers):
                h_next, c_next = self.cells[layer](x_t, (h[layer], c[layer]))
                x_t = h_next
                
                h_out.append(h_next)
                c_out.append(c_next)

            if not all(isinstance(item, torch.Tensor) for item in h_out):
                raise TypeError("h_out contains non-Tensor elements")
            if not all(isinstance(item, torch.Tensor) for item in c_out):
                raise TypeError("c_out contains non-Tensor elements")

            h, c = torch.stack(h_out), torch.stack(c_out)
        
        return h[-1], (h, c)
