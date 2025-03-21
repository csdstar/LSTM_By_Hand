import re

def sentence_tokenize(text):
    # 使用正则表达式按句号、问号、感叹号进行句子分割
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [sent.strip() for sent in sentences if sent]

def word_tokenize(sentence):
    # 将句子按空格、标点符号分割成单词
    words = re.findall(r'\b\w+\b', sentence)
    return words
