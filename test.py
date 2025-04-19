import csv
import os
import warnings

import faiss
import jieba
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from LSTM_self import LSTM, check_all_gradients

# 禁用 FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)

embedding_pt_path = "./data/word_embeddings.pt"
word2index_pt_path = "./data/word2index.pt"
faiss_index_path = "./data/faiss_index_file.index"

max_len = 50  # 设定最大序列长度
batch_size = 32  # 设定批量大小

# 生成训练数据：重复句子对 "我 爱 吃 苹果" -> "爱 吃 苹果 。"
input_texts = ["我爱吃雪梨"] * 3000
label_texts = ["爱吃雪梨。"] * 3000
test_input_texts = ["我爱吃雪梨"] * 32
test_label_texts = ["爱吃雪梨。"] * 32

words_num = 636013
words_dim = 300
h_list = [3, 9, 81]

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


# 构建 Faiss 索引所需的数据
def build_faiss_index(embedding_dict, word2index_dict):
    index2word = [word for word, _ in sorted(word2index_dict.items(), key=lambda x: x[1])]
    # 如果索引文件已存在，则直接加载
    if os.path.exists(faiss_index_path):
        print(f"Loading Faiss index from {faiss_index_path}...")
        index = faiss.read_index(faiss_index_path)
    else:
        embedding_dim = next(iter(embedding_dict.values())).shape[0]
        embedding_matrix = torch.stack([embedding_dict[word] for word in index2word])
        embedding_matrix_np = embedding_matrix.numpy().astype('float32')
        # 构建索引（L2 距离）
        print(f"Building Faiss index with {len(index2word)} vectors...")
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embedding_matrix_np)

        # 保存索引
        print(f"Saving Faiss index to {faiss_index_path}...")
        faiss.write_index(index, faiss_index_path)

    return index, index2word


def query_nearest_words(faiss_index, index2word, Y_pred_tensor, top_k=1):
    # Y_pred_tensor: (batch, embed_dim)
    Y_pred_np = Y_pred_tensor.detach().cpu().numpy().astype('float32')

    # 查询 top_k 个最相似的词
    D, I = faiss_index.search(Y_pred_np, top_k)  # D 是距离，I 是索引
    predicted_words = [[index2word[idx] for idx in row] for row in I]
    return predicted_words


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


def train():
    # 构造训练数据,转换为词向量(batch, seq_len, embed_dim)
    input_sentences = input_texts
    input_tokenized = tokenize_sentences(input_sentences, embedding_dict)
    inputs = tokens_to_tensor(input_tokenized, embedding_dict)

    # 标签数据，同样将其转为词向量(batch, seq_len, embed_dim)
    label_sentences = label_texts
    label_tokenized = tokenize_sentences(label_sentences, embedding_dict)
    labels = tokens_to_tensor(label_tokenized, embedding_dict)

    # 构建数据集
    dataset = TensorDataset(inputs, labels)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = LSTM(batch_size, words_dim, words_dim, h_list.copy())
    if os.path.exists("./model.pth"):
        model.load_state_dict(torch.load("./model.pth"))  # 加载权重

    model.train()

    losses = []
    total_loss = 0
    loss_func = torch.nn.MSELoss(reduction='mean')

    for epoch in range(25):
        print("Epoch: ", epoch)
        for X_batch, Y_batch in tqdm(loader):
            # 从 (batch, seq_len, input_dim) → (seq_len, batch, input_dim)
            X_batch = X_batch.permute(1, 0, 2)

            # 初始化总损失与每个时间步的Y梯度列表
            total_loss = 0
            dY_list = []

            # 进行前向传播
            model.forward(X_batch)
            seq_len = len(model.Y_list)

            # 逐时间步计算损失并累加
            for t in range(seq_len):
                Y_pred = model.Y_list[t]  # (batch, out_dim)
                Y_true = Y_batch[:, t]  # (batch,)

                # 累加损失：每个样本一个数，先平均再加权
                loss_t = loss_func(Y_pred, Y_true)
                total_loss += loss_t

                # 计算每个时间步的dL/dY
                with torch.no_grad():
                    # 计算当前时间步的梯度
                    dY = 2 * (Y_pred - Y_true) / (batch_size * words_dim)
                    dY_list.append(dY)

            # 反向传播, 包含zero和step
            model.backward(dY_list)

            # 打印当前所有梯度的最大值和分布
            check_all_gradients(model)

            # 清理缓存
            model.clear_memory()

        losses.append(total_loss)
        print(f"epoch {epoch} total_loss:{total_loss}")

    # 绘制损失曲线
    plt.plot(range(len(losses)), losses)  # x 轴为 epoch，y 轴为损失
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), "./model.pth")


def test():
    faiss_index, index2word = build_faiss_index(embedding_dict, word2index_dict)

    # 测试数据预处理
    test_sentences = test_input_texts  # 测试原始句子列表
    test_tokenized = tokenize_sentences(test_sentences, embedding_dict)
    test_inputs = tokens_to_tensor(test_tokenized, embedding_dict)  # (batch, seq_len, dim)

    # 测试标签（如果需要计算准确率等）
    label_sentences = test_label_texts
    label_tokenized = tokenize_sentences(label_sentences, embedding_dict)
    test_labels = tokens_to_tensor(label_tokenized, embedding_dict)

    # 构建测试集
    dataset = TensorDataset(test_inputs, test_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 模型参数
    model = LSTM(batch_size, words_dim, words_dim, h_list.copy())
    print(model)
    model.load_state_dict(torch.load("./model.pth"))  # 加载权重
    model.eval()  # 切换到评估模式（禁用 dropout）

    with torch.no_grad():  # 不计算梯度，节省内存
        for X_batch, Y_batch in loader:
            X_batch = X_batch.permute(1, 0, 2)  # (batch, seq_len, dim) → (seq_len, batch, dim)

            Y_pred = model.forward(X_batch)  # (batch, out_dim)
            Y_pred_np = Y_pred.cpu().detach().numpy().astype('float32')  # 转为 numpy 格式

            # 使用 Faiss 查找每一行向量最近的一个词
            D, I = faiss_index.search(Y_pred_np, k=1)  # D: 距离，I: 索引

            # 映射回词
            predicted_words = [index2word[idx] for idx in I[:, 0]]

            print(f"预测向量为: {Y_pred}")
            print(f"标签向量为: {Y_batch}")
            print(f"预测词为: {predicted_words}")  # 输出预测词


def main():
    # train()
    test()


if __name__ == "__main__":
    main()
