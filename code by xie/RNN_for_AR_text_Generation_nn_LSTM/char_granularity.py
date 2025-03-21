import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Load and preprocess text data
with open('corpus/Richard.txt', encoding='utf-8') as f:
    raw_text = f.read().lower()

chars = sorted(list(set(raw_text)))
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

print(chars)
print(len(raw_text))

# Define sequence length and create training data
seq_length = 100
x = []
y = []

for i in range(len(raw_text) - seq_length):
    given = raw_text[i:i + seq_length]
    predict = raw_text[i + seq_length]
    x.append([char_to_int[char] for char in given])
    y.append(char_to_int[predict])

print(x[:3])
print(y[:3])

# Define vocabulary size and reshape input data
n_patterns = len(x)
n_vocab = len(chars)

x = np.array(x)
x = np.reshape(x, (n_patterns, seq_length, 1))
x = x / float(n_vocab)
y = np.array(y)

# Convert data to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Define LSTM model in PyTorch
class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Take only the last time step
        out = self.fc(out)
        return out

# Set model parameters
input_size = 1
hidden_size = 128
output_size = n_vocab

model = CharLSTM(input_size, hidden_size, output_size)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 10
batch_size = 32

dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# # Save the model
# torch.save(model.state_dict(), 'my_model.pth')

# Prediction utilities
def predict_next(input_array):
    input_array = np.array(list(input_array))
    x = torch.tensor(input_array, dtype=torch.float32).view(1, seq_length, 1)
    x = x / float(n_vocab)
    with torch.no_grad():
        output = model(x)
        _, predicted_idx = torch.max(output, dim=1)
    return predicted_idx.item()

def string_to_index(raw_input):
    res = [char_to_int[c] for c in raw_input]
    
    if len(res) < seq_length:
        res = [0] * (seq_length - len(res)) + res
    
    return res[-seq_length:]

def y_to_char(y_idx):
    return int_to_char[y_idx]

def generate_article(init, rounds=500):
    in_string = init.lower()
    for _ in range(rounds):
        input_seq = string_to_index(in_string)
        next_index = predict_next(input_seq)
        in_string += y_to_char(next_index)
    return in_string

# Generate text using the model
init = 'To be or not to be, that is the question...'
article = generate_article(init)
print(article)
