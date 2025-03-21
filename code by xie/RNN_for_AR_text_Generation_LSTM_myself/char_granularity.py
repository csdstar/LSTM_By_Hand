import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import Counter

# Read and preprocess the text
raw_text = ''
for file in os.listdir("corpus"):
    if file.endswith(".txt"):
        with open("corpus/" + file, errors='ignore', encoding='utf-8') as f:
            raw_text += f.read() + '\n\n'

raw_text = raw_text.lower()
chars = list(set(raw_text))
print(f"Unique characters: {len(chars)}")

# Create a character-to-index mapping and index-to-character mapping
char2idx = {ch: idx for idx, ch in enumerate(chars)}
idx2char = {idx: ch for idx, ch in enumerate(chars)}

# Convert text to a sequence of character indices
text_stream = [char2idx[char] for char in raw_text if char in char2idx]

# Prepare sequences for LSTM with categorical labels
seq_length = 10
x_data, y_data = [], []
for i in range(0, len(text_stream) - seq_length):
    given = text_stream[i:i + seq_length]
    predict = text_stream[i + seq_length]
    x_data.append(given)
    y_data.append(predict)  # Categorical label

x_data = np.array(x_data)
y_data = np.array(y_data)

# Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.long)
        self.y_data = torch.tensor(y_data, dtype=torch.long)  # For cross-entropy

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

dataset = TextDataset(x_data, y_data)
dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)

# Define the character-level LSTM model
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256, num_layers=2):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = out[:, -1, :]  # Take the last output of the sequence
        out = self.fc(out)
        return out

# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharLSTM(vocab_size=len(chars), embedding_dim=128, hidden_size=256, num_layers=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Prediction function with temperature, top-k, and top-p sampling
def predict_next(input_seq, temperature=1.2, top_k=8, top_p=1.0):
    input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        y = model(input_seq)
    
    y = y / temperature
    probabilities = torch.softmax(y, dim=-1).cpu().numpy().flatten()

    sorted_indices = np.argsort(probabilities)[-top_k:]
    sorted_probs = probabilities[sorted_indices]
    
    cumulative_probs = np.cumsum(sorted_probs / sorted_probs.sum())
    cutoff_index = np.searchsorted(cumulative_probs, top_p)
    selected_indices = sorted_indices[-cutoff_index:]

    filtered_probs = probabilities[selected_indices]
    filtered_probs /= filtered_probs.sum()  # Normalize probabilities
    predicted_index = np.random.choice(selected_indices, p=filtered_probs)
    
    return predicted_index

def generate_text(init, rounds=100):
    input_seq = [char2idx[ch] for ch in init.lower()[-seq_length:] if ch in char2idx]
    generated_text = init
    
    for _ in range(rounds):
        predicted_index = predict_next(input_seq)
        next_char = idx2char[predicted_index]
        
        generated_text += next_char
        input_seq.append(predicted_index)
        
        if len(input_seq) > seq_length:
            input_seq = input_seq[1:]
    
    return generated_text

# Generate a character-level sequence
init_text = "To be, or not to be"
generated_text = generate_text(init_text)
print(generated_text)
