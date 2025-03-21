import os
import numpy as np
import torch
import torch.nn as nn
import random
from gensim.models.word2vec import Word2Vec
from torch.utils.data import DataLoader, Dataset
from tokenizer import sentence_tokenize, word_tokenize
from LSTM import MYLSTM
from sigmoid import CustomSigmoid
from cross_entropy import CustomCrossEntropyLoss

# Read and preprocess the text
raw_text = ''
for file in os.listdir("corpus"):
    if file.endswith(".txt"):
        with open("corpus/" + file, errors='ignore', encoding='utf-8') as f:
            raw_text += f.read() + '\n\n'

raw_text = raw_text.lower()

sents = sentence_tokenize(raw_text)
corpus = [word_tokenize(sen) for sen in sents]

print(len(corpus))
print(corpus[:3])

# Train Word2Vec model
w2v_model = Word2Vec(corpus, vector_size=128, window=5, min_count=5, workers=4)

# Flatten corpus for sequence generation
raw_input = [item for sublist in corpus for item in sublist]

# Filter out unknown words
vocab = w2v_model.wv.key_to_index
text_stream = [word for word in raw_input if word in vocab]

# Prepare sequences for LSTM with categorical labels
seq_length = 10
x_data, y_data = [], []
for i in range(0, len(text_stream) - seq_length):
    given = text_stream[i:i + seq_length]
    predict = text_stream[i + seq_length]
    x_data.append([w2v_model.wv[word] for word in given])
    y_data.append(vocab[predict])  # Change to vocab index for cross-entropy

x_data = np.array(x_data)
y_data = np.array(y_data)

class TextDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.long)  # For cross-entropy

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

dataset = TextDataset(x_data, y_data)
dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)

class TextLSTM(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, output_size=None, seq_length=10, num_layers=2):
        super(TextLSTM, self).__init__()
        self.lstm = MYLSTM(input_size, hidden_size, num_layers=num_layers)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)  # Output size should be vocab size
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        if out.dim() == 3:
            out = out[:, -1, :]
        out = self.fc(out)
        return self.softmax(out)

# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextLSTM(input_size=128, hidden_size=256, output_size=len(vocab), seq_length=10, num_layers=2).to(device)
criterion = CustomCrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# def predict_next(input_array):
#     if len(input_array) < seq_length:
#         padding = np.zeros((seq_length - len(input_array), 128), dtype=np.float32)
#         input_array = np.concatenate((padding, np.array(input_array)), axis=0)
#     elif len(input_array) > seq_length:
#         input_array = input_array[-seq_length:]

#     input_array = np.array(input_array, dtype=np.float32)
#     input_tensor = torch.tensor(input_array, dtype=torch.float32).to(device)
#     input_tensor = input_tensor.view(-1, seq_length, 128)

#     with torch.no_grad():
#         y = model(input_tensor)
    
#     # Get the word index with the highest probability
#     predicted_index = torch.argmax(y, dim=1).item()
#     return predicted_index

def predict_next(input_array, temperature=2.0, top_k=8, top_p=1.0):
    if len(input_array) < seq_length:
        padding = np.zeros((seq_length - len(input_array), 128), dtype=np.float32)
        input_array = np.concatenate((padding, np.array(input_array)), axis=0)
    elif len(input_array) > seq_length:
        input_array = input_array[-seq_length:]

    input_array = np.array(input_array, dtype=np.float32)
    input_tensor = torch.tensor(input_array, dtype=torch.float32).to(device)
    input_tensor = input_tensor.view(-1, seq_length, 128)

    with torch.no_grad():
        y = model(input_tensor)

    # Apply temperature scaling
    y = y / temperature
    probabilities = torch.softmax(y, dim=1).cpu().numpy().flatten()

    # Top-K filtering
    sorted_indices = np.argsort(probabilities)[-top_k:]
    sorted_probs = probabilities[sorted_indices]
    
    # Top-P (nucleus) filtering
    cumulative_probs = np.cumsum(sorted_probs / sorted_probs.sum())
    cutoff_index = np.searchsorted(cumulative_probs, top_p)
    selected_indices = sorted_indices[-cutoff_index:]

    # Sample from filtered probabilities
    filtered_probs = probabilities[selected_indices]
    filtered_probs /= filtered_probs.sum()  # Normalize probabilities
    predicted_index = np.random.choice(selected_indices, p=filtered_probs)
    
    return predicted_index


def string_to_index(raw_input):
    raw_input = raw_input.lower()
    input_stream = word_tokenize(raw_input)
    return [w2v_model.wv[word] for word in input_stream[-seq_length:] if word in vocab]

def y_to_word(predicted_index):
    return w2v_model.wv.index_to_key[predicted_index]  # Convert index back to word

def generate_article(init, rounds=30):
    in_string = init.lower()
    for _ in range(rounds):
        input_seq = string_to_index(in_string)
        predicted_index = predict_next(input_seq)
        next_word = y_to_word(predicted_index)
        
        if next_word:
            in_string += ' ' + next_word
        else:
            break
    return in_string

# Generate an article
init = "By accident most strange, bountiful Fortune, Now my dear lady, hath mine enemies"
article = generate_article(init)
print(article)
