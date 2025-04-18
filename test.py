import csv
import warnings

import jieba
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

from LSTM_self import LSTM

# 禁用 FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)

embedding_pt_path = "./data/word_embeddings.pt"
word2index_pt_path = "./data/word2index.pt"

max_len = 50  # 设定最大序列长度
batch_size = 32  # 设定批量大小

# 生成训练数据：重复句子对 "我 爱 吃 苹果" -> "爱 吃 苹果 。"
input_texts = ["我爱吃苹果"] * 300
label_texts = ["爱吃苹果。"] * 300

WORDS_NUM = 636013
WORDS_DIM = 300


def fake_data_test():
    # 假设一个批次的样本数为 3，序列长度为 7，输入特征维度为 10
    batch_size = 7
    sequence_length = 8
    in_dimension = 10
    out_dimension = 20

    # 随机生成输入数据（假数据）
    X = torch.randn(sequence_length, batch_size, in_dimension)  # (seq_len, batch_size, in_dimension)
    Y_true = torch.randint(0, out_dimension, (batch_size,), dtype=torch.long)  # 每个样本一个类别

    model = LSTM(batch_size, in_dimension, out_dimension, [3, 4])
    model.train()

    losses = []

    for epoch in range(150):
        # 传入模型进行前向传播
        Y = model.forward(X)

        # # 打印输出，检查是否正常运行
        # print(f"输出Y为:{Y}")

        # 计算交叉熵损失和dY，进行反向传播
        softmax_output = torch.softmax(Y, dim=1)  # (batch_size, vocab_size)
        eps = 1e-9
        log_probs = torch.log(softmax_output + eps)  # 防止 log(0)
        Y_one_hot = torch.nn.functional.one_hot(Y_true, num_classes=Y.shape[1]).float()

        loss = - (Y_one_hot * log_probs).sum(dim=1).mean()
        print("Epoch: {}, Loss: {}".format(epoch, loss))
        losses.append(loss.item())

        dY = softmax_output - Y_one_hot
        model.backward(dY)

    # 绘制损失曲线
    plt.plot(range(len(losses)), losses)  # x 轴为 epoch，y 轴为损失
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()


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


def train():
    embedding_dict, word2index_dict = load_embedding_from_pt()
    print("embedding dict loaded")

    # 构造训练数据,转换为词向量(batch, seq_len, embedding_vector)
    input_sentences = input_texts
    input_tokenized = tokenize_sentences(input_sentences, embedding_dict)
    inputs = tokens_to_tensor(input_tokenized, embedding_dict)

    # 标签数据，将其转为词索引(batch, seq_len, )
    label_sentences = label_texts
    label_tokenized = tokenize_sentences(label_sentences, embedding_dict)
    label_indices = [words_to_indices(tokens, word2index_dict) for tokens in label_tokenized]
    print(label_indices[0])
    labels = torch.tensor(label_indices, dtype=torch.long)

    # 构建数据集
    dataset = TensorDataset(inputs, labels)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = LSTM(batch_size, 300, WORDS_NUM, [3, 4])

    model.train()

    losses = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")  # 每个时间步独立计算

    for epoch in range(10):
        print("Epoch: ", epoch)
        for X_batch, Y_batch in loader:
            # 从 (batch, seq_len, input_dim) → (seq_len, batch, input_dim)
            X_batch = X_batch.permute(1, 0, 2)

            # 初始化总损失与每个时间步的Y梯度列表
            total_loss = 0
            dY_list = []

            # 进行前向传播
            model.forward(X_batch)
            seq_len = len(model.Y_list)

            # 逐时间步计算交叉熵损失并累加
            for t in range(seq_len):
                Y_pred = model.Y_list[t]  # (batch, vocab_size)
                Y_true = Y_batch[:, t]  # (batch,)

                # 累加交叉熵损失：每个样本一个数，先平均再加权
                loss_t = loss_fn(Y_pred, Y_true).mean()
                total_loss += loss_t

                # Softmax 导出梯度（softmax - onehot）
                softmax_out = torch.softmax(Y_pred, dim=1)
                softmax_out[range(Y_pred.size(0)), Y_true] -= 1
                dY_list.append(softmax_out)  # 加入列表，稍后反向传播

            losses.append(total_loss)
            print(f"epoch {epoch} total_loss:{total_loss}")

            # 反向传播
            model.backward(dY_list)

            model.clear_memory()

    # 绘制损失曲线
    plt.plot(range(len(losses)), losses)  # x 轴为 epoch，y 轴为损失
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), "model.pth")


def test(model):
    pass


def main():
    train()


if __name__ == "__main__":
    main()
