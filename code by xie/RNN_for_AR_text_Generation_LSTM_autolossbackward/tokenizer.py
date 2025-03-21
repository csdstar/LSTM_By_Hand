import re

def sentence_tokenize(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [sent.strip() for sent in sentences if sent]

def word_tokenize(sentence):
    words = re.findall(r'\b\w+\b', sentence)
    return words
