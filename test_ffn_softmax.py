import csv
import warnings

import jieba
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from LSTM_self import LSTM
from ffn import FFN_in, FFN_out

# 禁用 FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)

device_cpu = torch.device('cpu')
device_gpu = torch.device('cuda')

embedding_pt_path = "./data/word_embeddings.pt"
word2index_pt_path = "./data/word2index.pt"
index2word_pt_path = "./data/index2word.pt"
faiss_index_path = "./data/faiss_index_file.index"

max_len = 50  # 设定最大序列长度
batch_size = 64  # 设定批量大小

# 生成训练数据：重复句子对
input_texts = ["我爱吃苹果"] * 3000
label_texts = ["爱吃苹果。"] * 3000
test_input_texts = ["我爱吃苹果"] * 32
test_label_texts = ["爱吃苹果。"] * 32

words_num = 636013
embed_dim = 300
ffn_in_hidden_size = 512
lstm_outdim = 768
h_list = [512, 512, 768]

# 加载词向量和映射
embedding_dict, word2index_dict, index2word_dict = torch.load(embedding_pt_path), torch.load(word2index_pt_path), torch.load(index2word_pt_path)
print("dict loaded")


# 加载词嵌入向量字典
def load_embedding_from_csv(filepath='./data/word_embeddings.csv'):
    # 词向量字典 word -> vector
    embedding_dict = {}

    # 词索引字典 word -> index
    word2index = {}

    # 索引字典  index -> word
    index2word = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        next(reader)  # 跳过无效行

        for index, row in enumerate(reader):
            word = row[0]
            try:
                vector = torch.tensor([float(x) for x in row[1:]], dtype=torch.float32)
                embedding_dict[word] = vector
                word2index[word] = index
                index2word[index] = word
            except ValueError:
                continue

    # 保存三个文件
    torch.save(embedding_dict, embedding_pt_path)
    print(f"✅ 词→向量字典已保存至: {embedding_pt_path}")

    torch.save(word2index, word2index_pt_path)
    print(f"✅ 词→索引字典已保存至: {word2index_pt_path}")

    torch.save(index2word, index2word_pt_path)
    print(f"✅ 索引→词字典已保存至: {index2word_pt_path}")


# 从保存的pt快照中加载三个字典
def load_embedding_from_pt():
    return torch.load(embedding_pt_path), torch.load(word2index_pt_path), torch.load(index2word_pt_path)


# 将词语列表转换为对应字典索引列表
def words_to_indices(words, word2index_dict):
    indices = []
    for word in words:
        if word in word2index_dict:
            indices.append(word2index_dict[word])
        else:
            raise ValueError(f"标签词 '{word}' 不在词表中！")
    return indices


# 分词 + 过滤词典中不存在的词, 输入一个句子列表，返回token列表(sentence_num, seq_len)
def tokenize_sentences(sentences, embedding_dict):
    tokenized = []
    for sent in sentences:
        tokens = [t for t in jieba.lcut(sent) if t in embedding_dict]
        if tokens:
            tokenized.append(tokens)
    return tokenized


# 加载输入数据并分词
def load_and_tokenize(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    tokenized_lines = [jieba.lcut(line.strip()) for line in lines]
    return tokenized_lines


# 将tokens转为张量
def tokens_to_tensor(tokenized_data, embedding_dict, max_len=None):
    tensor_data = []
    for tokens in tokenized_data:
        vecs = []
        for token in tokens:
            if token in embedding_dict:
                vecs.append(embedding_dict[token])
            else:
                print("有OOV词！")
                continue  # 忽略OOV词

        if not vecs:
            print("整句都是OOV词，跳过")
            continue  # 如果没有任何词在词典中，跳过该句

        # 如果设置了最大序列长度，进行截断或填充
        if max_len:
            embedding_dim = next(iter(embedding_dict.values())).shape[0]
            if len(vecs) > max_len:
                vecs = vecs[:max_len]
            else:
                pad_num = max_len - len(vecs)
                vecs += [torch.zeros(embedding_dim)] * pad_num

        tensor_data.append(torch.stack(vecs))  # shape: (seq_len, embedding_dim)

    return torch.stack(tensor_data)  # shape: (batch, seq_len, embedding_dim)


# 训练函数
def train():
    # 构造训练数据,转换为词向量(batch, seq_len, embed_dim)
    input_sentences = input_texts
    input_tokenized = tokenize_sentences(input_sentences, embedding_dict)
    inputs = tokens_to_tensor(input_tokenized, embedding_dict)

    # 标签数据，同样将其转为词向量(batch, seq_len, embed_dim)
    label_sentences = label_texts
    label_tokenized = tokenize_sentences(label_sentences, embedding_dict)
    labels = torch.tensor([words_to_indices(words, word2index_dict) for words in label_tokenized], dtype=torch.long)

    # 构建数据集
    dataset = TensorDataset(inputs, labels)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = LSTM(batch_size, ffn_in_hidden_size, lstm_outdim, h_list.copy())
    ffn_in = FFN_in(embed_dim, ffn_in_hidden_size)
    ffn_out = FFN_out(lstm_outdim, words_num).to(device_gpu)

    model.train()
    ffn_in.train()
    ffn_out.train()

    losses = []
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(5):
        print("Epoch: ", epoch)
        batch_losses = []
        loss = 0
        for X_batch, Y_batch in tqdm(loader):
            # 初始化总损失与每个时间步的Y梯度列表
            batch_loss = 0

            # 进行前向传播
            X_batch = ffn_in(X_batch)
            # 从 (batch, seq_len, input_dim) → (seq_len, batch, input_dim)
            X_batch = X_batch.permute(1, 0, 2)
            model.forward(X_batch)
            seq_len = len(model.Y_list)

            # 清空上轮残留
            for y in model.Y_list:
                y.grad = None
                y.retain_grad()

            # 遍历每个时间步的输出 → FFN_out → 累加时间步损失 → 进行反向传播
            for t in range(seq_len):
                Y_lstm = model.Y_list[t]  # (batch, out_dim)

                # 将LSTM输出移动到GPU以进行FFN_out计算
                Y_pred = ffn_out(Y_lstm.to(device_gpu))  # (batch, words_num)
                Y_true = Y_batch[:, t].to(device_gpu)  # (batch)，值是索引

                batch_loss += loss_func(Y_pred, Y_true)

            # 自动反向传播：将梯度写入 Y_lstm.grad
            batch_loss.backward()

            # 收集梯度作为 BPTT 输入
            dY_list = [y.grad.detach().cpu() for y in model.Y_list]

            # 反向传播, 包含zero和step
            model.backward(dY_list)

            # 清理缓存，统计loss和batch_loss
            model.clear_memory()
            loss += batch_loss.detach()
            batch_losses.append(batch_loss.detach())

        # loss是整个epoch所有batch_loss之和
        print(f"epoch {epoch} total_loss:{loss}")
        losses.append(loss)

        # 在每个 epoch 绘制 batch_losses
        plt.plot(batch_losses.cpu().numpy())  # 绘制当前 epoch 中每个批次的损失
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title(f'Epoch {epoch} Batch Losses')
        plt.show()

    # 绘制损失曲线
    plt.plot(losses)  # x 轴为 epoch，y 轴为损失
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()

    # 保存模型
    torch.save({
        'lstm': model.state_dict(),
        'ffn_in': ffn_in.state_dict(),
        'ffn_out': ffn_out.state_dict()
    }, 'model.pth')


def test():
    # 测试数据预处理
    test_sentences = test_input_texts  # 测试原始句子列表
    test_tokenized = tokenize_sentences(test_sentences, embedding_dict)
    test_inputs = tokens_to_tensor(test_tokenized, embedding_dict)  # (batch, seq_len, dim)

    # 测试标签（如果需要计算准确率等）
    label_sentences = test_label_texts
    label_tokenized = tokenize_sentences(label_sentences, embedding_dict)
    test_labels = torch.tensor([words_to_indices(words, word2index_dict) for words in label_tokenized], dtype=torch.long)

    # 构建测试集
    dataset = TensorDataset(test_inputs, test_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 加载模型参数
    checkpoint = torch.load("./model.pth")
    print("Keys in checkpoint:", checkpoint.keys())
    model = LSTM(batch_size, ffn_in_hidden_size, lstm_outdim, h_list.copy())
    ffn_in = FFN_in(embed_dim, ffn_in_hidden_size)
    ffn_out = FFN_out(lstm_outdim, words_num)

    ffn_in.load_state_dict(checkpoint['ffn_in'])
    ffn_out.load_state_dict(checkpoint['ffn_out'])
    model.load_state_dict(checkpoint['lstm'])

    # 切换到评估模式（禁用 dropout）
    ffn_in.eval()
    ffn_out.eval()
    model.eval()

    with torch.no_grad():  # 不计算梯度，节省内存
        for X_batch, Y_batch in loader:
            X_batch = ffn_in(X_batch)
            X_batch = X_batch.permute(1, 0, 2)
            model.forward(X_batch)
            seq_len = len(model.Y_list)
            for t in range(seq_len):
                Y_lstm = model.Y_list[t]
                Y_pred_logits = ffn_out(Y_lstm.detach())  # (batch, words_num)

                # 使用logits从词表中选择最大概率词
                Y_pred_index = torch.argmax(Y_pred_logits, dim=1)[0]  # (batch,)
                Y_true_index = Y_batch[0][t]

                pred_word = index2word_dict[Y_pred_index.item()]
                true_word = index2word_dict[Y_true_index.item()]

                print(f"时间步{t}: 真实词:{true_word} 预测词:{pred_word}")


def main():
    train()
    test()


if __name__ == "__main__":
    main()
