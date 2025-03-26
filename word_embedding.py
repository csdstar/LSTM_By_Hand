import bz2
import shutil

import chardet
import pandas as pd

input_file = 'sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.bz2'
output_file = 'sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.txt'

input_txt = 'sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.txt'
output_csv = 'word_embeddings.csv'


def detect_char():
    # 检测文件编码
    with open(input_txt, 'rb') as f:
        result = chardet.detect(f.read(10000))  # 读取文件的前 10000 字节进行检测

    print(f"检测到的编码格式为: {result['encoding']}")


def unzip():
    with bz2.BZ2File(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"解压完成，输出文件为: {output_file}")


def to_csv():
    # 逐行读取数据并转换
    data = []
    with open(input_txt, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]  # 单词
            vector = list(map(float, parts[1:]))  # 向量
            data.append([word] + vector)  # 每行数据：[词, d1, d2, ..., d300]

    # 创建 DataFrame
    columns = ['word'] + [f'dim_{i}' for i in range(1, 301)]
    df = pd.DataFrame(data, columns=columns)

    # 将 DataFrame 写入 CSV 文件
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"已成功将数据转换为 CSV 文件：{output_csv}")


def main():
    to_csv()


if __name__ == '__main__':
    main()
