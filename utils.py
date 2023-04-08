import collections
import pickle
import numpy as np

import pandas as pd
from tokenizer import Tokenizer
from gensim.models import Word2Vec
import re

# from gensim.test.utils import get_tmpfile

train_article_path = "sumdata/train/train.article.txt"
train_title_path = "sumdata/train/train.title.txt"
valid_article_path = "sumdata/train/valid.article.filter.txt"
valid_title_path = "sumdata/train/valid.title.filter.txt"

# train_article_path = "sumdata/train/training_article.txt"
# train_title_path = "sumdata/train/traning_title.txt"

def clean_str(text):
    try:
        text = re.sub(
            r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?', '', text)
        text = re.sub(r'[|:}{=]', ' ', text)
        text = re.sub(r'[;]', ' ', text)
        text = re.sub(r'[\n]', ' ', text)
        text = re.sub(r'[\t]', ' ', text)
        text = re.sub(r'[-]', ' ', text)
        text = re.sub(r'[+]', ' ', text)
        text = re.sub(r'[*]', ' ', text)
        text = re.sub(r'[/]', ' ', text)
        text = re.sub(r'[//]', ' ', text)
        text = re.sub(r'[@]', ' ', text)
        text = re.sub(r'[,]', ' ', text)
        text = re.sub(r'[)]', ' ', text)
        text = re.sub(' +', ' ', text)
        # text = re.sub('\n+', '\n', text)
        # text = re.sub('\t+', '\t', text)
        # text = re.sub('\n+', '\n', text)
        text = re.sub(r'[-]', ' ', text)
        text = re.sub(r'[(]', ' ', text)
        text = re.sub(' + ', ' ', text)
        # text = text.encode('ascii', errors='ignore').decode("utf-8")
        return text
    except Exception as e:
        print(e)
        print(f"Error while cleaning text --->{text}")
        pass

def get_text_list(data_path, toy):
    with open (data_path, "r", encoding="utf-8") as f:
        if not toy:
            val = [clean_str(" ".join(x.split())) for x in f.readlines()][:10000]
            val = [x for x in val if x]
            return val
        else:
            val = [clean_str(" ".join(x.split())) for x in f.readlines()][:1000]
            val = [x for x in val if x]
            return val

def build_dict(step, toy=False):
    tokenizer = Tokenizer()
    if step == "train":
        train_article_list = get_text_list(train_article_path, toy)
        train_title_list = get_text_list(train_title_path, toy)

        words = list()
        for sentence in train_article_list + train_title_list:
            for word in tokenizer.word_tokenize(sentence):
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<padding>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<s>"] = 2
        word_dict["</s>"] = 3
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)

        with open("word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    elif step == "valid":
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))

    article_max_len = 50
    summary_max_len = 15

    return word_dict, reversed_dict, article_max_len, summary_max_len

def build_dataset(step, word_dict, article_max_len, summary_max_len, toy=False):
    if step == "train":
        article_list = get_text_list(train_article_path, toy)
        title_list = get_text_list(train_title_path, toy)
    elif step == "valid":
        article_list = get_text_list(valid_article_path, toy)
    else:
        raise NotImplementedError
    tokenizer = Tokenizer()
    x = [tokenizer.word_tokenize(d) for d in article_list]
    x = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in x]
    x = [d[:article_max_len] for d in x]
    x = [d + (article_max_len - len(d)) * [word_dict["<padding>"]] for d in x]
    
    if step == "valid":
        return x
    else:        
        y = [tokenizer.word_tokenize(d) for d in title_list]
        y = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in y]
        y = [d[:(summary_max_len - 1)] for d in y]
        return x, y

def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]


def get_init_embedding(reversed_dict, embedding_size):
    word2vec_file = "word2vec_model/new_sabda_to_vec_model_md"
    print("Loading Word2vec vectors...")
    word_vectors = Word2Vec.load(word2vec_file)
    print("Word2vec vectors loaded...")
    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors.wv[word]
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)
            print(f"Couldn't embbed : {word}")

        word_vec_list.append(word_vec)

    # Assign random vector to <s>, </s> token
    word_vec_list[2] = np.random.normal(0, 1, embedding_size)
    word_vec_list[3] = np.random.normal(0, 1, embedding_size)

    return np.array(word_vec_list)