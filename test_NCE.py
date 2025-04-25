import csv
import os
import warnings

import jieba
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from LSTM_self import LSTM
from NCE import NCELoss

# 禁用 FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)

embedding_pt_path = "./data/word_embeddings.pt"
word2index_pt_path = "./data/word2index.pt"

max_len = 50  # 设定最大序列长度
batch_size = 32  # 设定批量大小

# 生成训练数据：重复句子对 "我 爱 吃 苹果" -> "爱 吃 苹果 。"
input_texts = ["我爱吃雪梨"] * 3000
label_texts = ["爱吃雪梨。"] * 3000
test_input_texts = ["我爱吃雪梨"] * 32
test_label_texts = ["爱吃雪梨。"] * 32

words_num = 636013
embed_dim = 300
h_list = [30, 120, 300]

# 加载词向量和映射
embedding_dict, word2index_dict = torch.load(embedding_pt_path), torch.load(word2index_pt_path)
print("embedding dict loaded")


# 加载词嵌入向量字典
def load_embedding_from_csv(filepath):
    # 词向量字典
    embedding_dict = {}

    # 词索引字典
    word2index = {}

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
            except ValueError:
                continue
    torch.save(embedding_dict, embedding_pt_path)
    print(f"词向量已保存至 {embedding_pt_path}")

    torch.save(word2index, word2index_pt_path)
    print(f"词索引已保存至 {word2index_pt_path}")


# 从保存的pt快照中加载字典
def load_embedding_from_pt():
    return torch.load(embedding_pt_path), torch.load(word2index_pt_path)


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

    # 标签数据，将其转为词索引: 形状是(batch, seq_len)，每个值都是索引
    label_sentences = label_texts
    label_tokenized = tokenize_sentences(label_sentences, embedding_dict)
    labels = []
    for sentence in label_tokenized:
        labels.append(words_to_indices(sentence, word2index_dict))  # list of list of int
    labels = torch.tensor(labels, dtype=torch.long)

    # 构建数据集
    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 初始化词嵌入层
    embedding_tensor = torch.stack([embedding_dict[word] for word in word2index_dict.keys()])
    embedding_layer = torch.nn.Embedding.from_pretrained(embedding_tensor, freeze=False)
    print("词嵌入层初始化完毕")

    model = LSTM(batch_size, embed_dim, embed_dim, h_list.copy())
    if os.path.exists("./model.pth"):
        model.load_state_dict(torch.load("./model.pth"))  # 加载权重

    model.train()

    nce_loss_fn = NCELoss(embedding_layer.weight, num_sampled=20)

    losses = []

    for epoch in range(10):
        print("Epoch: ", epoch)
        loss = 0
        batch_losses = []
        for X_batch, Y_batch in tqdm(loader):
            # 从 (batch, seq_len, input_dim) → (seq_len, batch, input_dim)
            X_batch = X_batch.permute(1, 0, 2)
            # 从(batch, seq_len) -> (seq_len, batch)
            Y_batch = Y_batch.permute(1, 0)

            # 初始化每个时间步的Y梯度列表
            dY_list = []

            # 进行前向传播
            model.forward(X_batch)
            seq_len = len(model.Y_list)
            batch_loss = 0

            for t in range(seq_len):
                y_pred = model.Y_list[t]  # [batch, embed_dim]
                y_true = Y_batch[t]  # [batch]
                loss_t, dY = nce_loss_fn(y_pred, y_true)
                batch_loss += loss_t

            # 反向传播, 包含zero和step
            model.backward(dY_list)

            # 清理缓存
            model.clear_memory()
            loss += batch_loss
            batch_losses.append(batch_loss)

        # 在每个 epoch 后绘制 batch_losses
        plt.plot(batch_losses)  # 绘制当前 epoch 中每个批次的损失
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title(f'Epoch {epoch} Batch Losses')
        plt.show()

        losses.append(loss)
        print(f"epoch {epoch} loss:{loss}")

    # 绘制总损失曲线
    plt.plot(losses)  # x 轴为 epoch，y 轴为损失
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), "./model.pth")


def test():
    # 测试数据预处理
    test_sentences = test_input_texts  # 测试原始句子列表
    test_tokenized = tokenize_sentences(test_sentences, embedding_dict)
    test_inputs = tokens_to_tensor(test_tokenized, embedding_dict)  # (batch, seq_len, dim)

    # 测试标签（如果需要计算准确率等）转为 index
    label_sentences = test_label_texts
    label_tokenized = tokenize_sentences(label_sentences, embedding_dict)
    test_labels = []
    for sentence in label_tokenized:
        test_labels.append(words_to_indices(sentence, word2index_dict))  # list of list of int
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # 构建测试集
    dataset = TensorDataset(test_inputs, test_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 模型参数
    model = LSTM(batch_size, embed_dim, embed_dim, h_list.copy())
    print(model)
    model.load_state_dict(torch.load("./model.pth"))  # 加载权重
    model.eval()  # 切换到评估模式（禁用 dropout）

    with torch.no_grad():  # 不计算梯度，节省内存
        for X_batch, Y_batch in loader:
            X_batch = X_batch.permute(1, 0, 2)  # (batch, seq_len, dim) → (seq_len, batch, dim)


def main():
    train()
    # test()


if __name__ == "__main__":
    main()
