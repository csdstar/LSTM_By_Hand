import os
import numpy as np
import torch
import torch.nn as nn
from gensim.models.word2vec import Word2Vec
from torch.utils.data import DataLoader, Dataset
from tokenizer import sentence_tokenize, word_tokenize
import random

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

# Prepare sequences for LSTM
seq_length = 10
x_data, y_data = [], []
for i in range(0, len(text_stream) - seq_length):
    given = text_stream[i:i + seq_length]
    predict = text_stream[i + seq_length]
    x_data.append([w2v_model.wv[word] for word in given])
    y_data.append(w2v_model.wv[predict])


x_data = np.array(x_data)
y_data = np.array(y_data)

# PyTorch Dataset
class TextDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.float32)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

dataset = TextDataset(x_data, y_data)
dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)

class TextLSTM(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, output_size=128, seq_length=10, num_layers=2):
        super(TextLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Only the last LSTM output is used
        out = torch.sigmoid(self.fc(out))
        return out

# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextLSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        # Forward pass
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def predict_next(input_array):
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
    
    return y.cpu().numpy()

def string_to_index(raw_input):
    raw_input = raw_input.lower()
    input_stream = word_tokenize(raw_input)
    return [w2v_model.wv[word] for word in input_stream[-seq_length:] if word in vocab]

def y_to_word(y, temperature=1.2, top_k=8):
    y = y.flatten()
    similar_words = w2v_model.wv.similar_by_vector(y, topn=top_k)
    
    if temperature != 1.0:
        probabilities = np.array([similar[1] for similar in similar_words])
        probabilities = np.exp(np.log(probabilities) / temperature)
        probabilities /= probabilities.sum()
    else:
        probabilities = [similar[1] for similar in similar_words]
    
    word_choices = [similar[0] for similar in similar_words]
    chosen_word = random.choices(word_choices, probabilities)[0]
    
    return chosen_word

def generate_article(init, rounds=30, temperature=1.2, top_k=8):
    in_string = init.lower()
    for _ in range(rounds):
        input_seq = string_to_index(in_string)
        next_word_vector = predict_next(input_seq)
        next_word = y_to_word(next_word_vector, temperature=temperature, top_k=top_k)
        
        if next_word:
            in_string += ' ' + next_word
        else:
            break
    return in_string

# Generate an article
init = "Heavens thank you for't! And now, I pray you, sir,"
article = generate_article(init)
print(article)
