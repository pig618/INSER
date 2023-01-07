import os
import random
import numpy as np
import torch
import tqdm
import re
import pickle
from datetime import datetime, timedelta
import pandas as pd
import csv
import collections
import nltk
from torch.utils.data import DataLoader
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
import sty

from helper.imdb import (
    LanguageIndex, binary_one_hot_convert, text2idx
)
import helper.generator as generator
import helper.encoder as encoder
import helper.generic as generic
import helper.train as train
import helper.utils as utils


def load_data(data, init_date, fin_date, ind_foc=('itech',)):    
    ind_foc = set(ind_foc)
    
    date_count = {}
    for i in range(len(data)):
        date = pd.to_datetime(data[i]["an"][8:16], format="%Y%m%d")
        if date >= init_date and date <= fin_date:
            ind_codes = data[i]["industry_codes"].split(",")
            
            ind_codes = set(ind_codes)            
            if len(ind_foc & ind_codes) > 0:            
            # if ind_foc in ind_codes:                
                if date in date_count:
                    date_count[date].append(i)
                else:
                    date_count[date] = [i]
    od = collections.OrderedDict(sorted(date_count.items()))
    return od, date_count


def load_market_data(path):
    market_return = pd.read_csv(path)
    market_return['date'] = pd.to_datetime(market_return['date'], format="%Y%m%d")
    sel_ret = market_return[market_return["date"] >= pd.to_datetime('01/01/14')][
        market_return["date"] <= pd.to_datetime('12/31/16')]
    return sel_ret[['date', 'mkt']]


def prepare_data(data, sel_ret, date_count):
    documents = []
    ky_ret = []
    ky_dates = []
    for i in range(len(sel_ret.iloc[:, 0])):
        dt = sel_ret.iloc[i, 0] - timedelta(days=1)
        if dt in date_count:
            for ar in date_count[dt]:
                ky_dates.append(sel_ret['date'].iloc[i])
                ky_ret.append(sel_ret['mkt'].iloc[i])
                if 'body' in data[ar]:
                    documents.append(data[ar]['art'] + data[ar]["body"])
                else:
                    documents.append(data[ar]['art'])
    return documents, ky_ret, ky_dates


def load_glove_embedding(fpath):
    word2embedding = {}
    with open(fpath, "r", errors='ignore') as f:
        for (i, line) in enumerate(f):
            data = line.strip().split(" ")
            word = data[0].strip()
            embedding = list(map(float, data[1:]))
            word2embedding[word] = np.array(
                embedding)  # shape -- (embedding_dim, )
            embedding_dim = len(embedding)

    return word2embedding, embedding_dim


def preprocess_documents(documents, word2embedding):
    for i in range(len(documents)):
        text = documents[i]
        text = re.sub('\W+', ' ', text).lower().strip()
        text = text.replace("\n", " ")
        text = nltk.word_tokenize(text)
        text = [word for word in text if not word in stop_words and 
                                not word.isnumeric()]
        text = [word for word in text if word in word2embedding]
        documents[i] = " ".join(text)


class rnnDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels, mask):
        self.inputs = inputs
        self.labels = labels
        self.mask = mask

    def __getitem__(self, idx):
        item = {}
        item['inputs'] = torch.tensor(self.inputs[idx])
        item['labels'] = torch.tensor(self.labels[idx])
        item['input_mask'] = torch.tensor(self.mask[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_train_test(documents, ky_ret, vocab):
    train_l = int(0.8*len(documents))
    test_l  = len(documents) - train_l
    max_seq_length = 300

    ### train
    x_tr = []
    m_tr = []
    y_tr = np.zeros(train_l, dtype = np.float32)

    ind = 0
    for example in range(train_l):
        x, m = text2idx(
            documents[example], max_seq_length, vocab.word2idx
        )
        x_tr.append(x)
        m_tr.append(m)
        y_tr[ind] = ky_ret[example]
        ind += 1

    x_tr = np.array(x_tr, dtype=np.int32)
    m_tr = np.array(m_tr, dtype=np.float32)
    print("x_tr shape: ", x_tr.shape)
    print("m_tr shape: ", m_tr.shape)

    ### test
    x_te = []
    m_te = []
    y_te = np.zeros(test_l, dtype = np.float32)

    ind = 0
    for example in range(train_l, train_l+test_l):
        x, m = text2idx(documents[example], max_seq_length, vocab.word2idx)
        x_te.append(x)
        m_te.append(m)
        y_te[ind] = ky_ret[example]
        ind += 1

    x_te = np.array(x_te, dtype=np.int32)
    m_te = np.array(m_te, dtype=np.float32)
    print("x_te shape: ", x_te.shape)
    print("m_te shape: ", m_te.shape)

    return x_tr, m_tr, y_tr, x_te, m_te, y_te
