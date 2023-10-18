
import random
import numpy as np
import torch.nn.functional as F
import torch
import os
import pandas as pd

def clean_tok_text(words):

    return [w.strip('Ġ') for w in words]

def get_max_len(sentences, tokenizer):
    max_len = 0 
    for sent in sentences:
        tok_sent = tokenizer.encode(sent, add_special_tokens=True)
        l = len(tok_sent)
        if l > max_len:
            max_len = l
    return max_len

def list_roles(info):
    role_list = []
    for i in info:
        role_list.append(i[0])
    return role_list

def read_file(path): # for old format only
    sentences = []
    target_words = []
    info = []
    with open(path, 'r') as rf:
        next(rf)
        for l in rf:
            line = l.strip().split('\t')
            sent = line[5].split(' ')
            if line[3] != 'NA':
                A = sent[int(line[3])]
                sentences.append(' '.join(sent).strip())
                target_words.append(A)
                info.append(('A', line[0], line[1]))
            if line[4] != 'NA':
                P = sent[int(line[4])]
                sentences.append(' '.join(sent).strip())
                target_words.append(P)
                info.append(('P', line[0], line[1]))

    return sentences, target_words, info

def read_data_splits(dir_path, scrambling=False):
    """
    Method used to read in data that is already split in train, valid and test
    :param dir_path: [String] The path to the directory that contains splits
    :param scrambling: [Boolean] True if words in sentences should be scrambled
    
    """
    files = os.listdir(dir_path)
    splits = dict()
    for f in files:
        if f.startswith('train'):
            file_path = os.path.join(dir_path, f)
            out = read_data(file_path, scrambling)
            splits['train'] = out
        elif f.startswith('valid'):
            file_path = os.path.join(dir_path, f)
            out = read_data(file_path, scrambling)
            splits['valid'] = out
        elif f.startswith('test'):
            file_path = os.path.join(dir_path, f)
            out = read_data(file_path, scrambling)
            splits['test'] = out
        else:
            continue
    return splits

def read_data_splits_ids(dir_path, label_column = 'role'):
    files = os.listdir(dir_path)
    splits = dict()
    for f in files:
        if f.endswith('train.txt'):
            file_path = os.path.join(dir_path, f)
            out = read_data_indices(file_path, label_column=label_column)
            splits['train'] = out

        elif f.endswith('test.txt'):
            file_path = os.path.join(dir_path, f)
            out = read_data_indices(file_path, label_column=label_column)
            splits['test'] = out
        elif f.endswith('val.txt'):
            file_path = os.path.join(dir_path, f)
            out = read_data_indices(file_path, label_column=label_column)
            splits['valid'] = out
        else:
            continue
    
    return splits
def read_data(path, scrambling=False, label_column = 'role', separ='\t'):
    df = pd.read_csv(path, sep=separ)
    if scrambling:
        print('not implemented yet!')
    return df['sent'], df['words'], df[label_column], df['ids']

"""
def read_data(path,scrambling=False):
    sentences = []
    target_words = []
    labels = []
    ids = []
    with open(path, 'r') as rf:
        next(rf)
        for l in rf:
            line = l.strip().split('\t')
            sent = line[2].strip()
            if scrambling:
                sent = sent.strip('.').split(' ')
                random.shuffle(sent)
                sent = [w.strip() for w in sent]
                sent = ' '.join(sent).strip() + ' .'

            sentences.append(sent)
            target_words.append(line[1].strip())
            labels.append(line[3].strip())
            ids.append(line[0].strip())
    return sentences, target_words, labels, ids
"""
def read_data_indices(path, label_column = 'role'):
    data = pd.read_csv(path, sep='\t')
    return data['sent_id'].to_list(), data['words'].to_list(), data['ids'].to_list(), data['sent'].to_list(), data[label_column].to_list()
    """
    sentences = []
    target_words = []
    target_ids = []
    labels = []
    ids = []
    with open(path, 'r') as rf:
        next(rf)
        for l in rf:
            line = l.strip().split('\t')
            #if scrambling: scrambling not possible here because indices wouldn't be correct
            ids.append(line[0].strip())
            target_words.append(line[1].strip())
            target_ids.append(line[2].strip())
            sentences.append(line[3].strip())
            labels.append(line[4].strip())
    return ids, target_words, target_ids, sentences, labels
    """

def get_splits(sentences, targets, info, train_size=0.7):
    sentences, targets, info = random_shuffle_three(sentences, targets, info)
    length_train = int(len(sentences) * train_size)
    train_sent = sentences[:length_train]
    train_target = targets[:length_train]
    train_inf = info[:length_train]
    valid_sent = sentences[length_train:]
    valid_target = targets[length_train:]
    valid_inf = info[length_train:]

    return (train_sent, train_target, train_inf), (valid_sent, valid_target, valid_inf)

def binary_class_cross_entropy(output, target):
    #assert output.shape == target.shape, "target shape is same as output shape"
    loss = F.binary_cross_entropy(output, target)
    return loss

def random_shuffle_three(one, two, three):
    c = list(zip(one, two, three))
    random.shuffle(c)
    one, two, three = zip(*c)
    #return zip(*c)
    return list(one), list(two), list(three)

def random_shuffle_five(one, two, three, four, five):
    c = list(zip(one, two, three, four,five))
    random.shuffle(c)
    one, two, three, four, five = zip(*c)
    #return zip(*c)
    return list(one), list(two), list(three), list(four), list(five)

def format_indices(ids_str):

    ids_str = str(ids_str).strip(',')
    ids_spl = ids_str.strip().split(',')

    return [int(id) for id in ids_spl]

def shorten_dataset(data, size):
    #ids, target_words, target_ids, sentences, labels 
    idcs,twords,tids,sents,labs =random_shuffle_five(data[0], data[1], data[2], data[3],data[4])
    ids = []
    target_ws = []
    target_is = []
    target_sents = []
    target_labels = []
    size = int(size/2)
    a = 0
    p = 0
  
    for i,tw,ti,s,l in zip(idcs,twords,tids,sents,labs):
        if a < size and p < size:
            ids.append(i)
            target_ws.append(tw)
            target_is.append(ti)
            target_sents.append(s)
            target_labels.append(l)
            if l == 'A':
                a+=1
            else:
                p+=1
        elif a < size and p >= size:
            if l == 'A':
                ids.append(i)
                target_ws.append(tw)
                target_is.append(ti)
                target_sents.append(s)
                target_labels.append(l)
                a+=1
            else:
                continue
        elif a >=size and p < size:
            if l == 'P':                
                ids.append(i)
                target_ws.append(tw)
                target_is.append(ti)
                target_sents.append(s)
                target_labels.append(l)
                p+=1
            else:
                continue
        else:
            break
    return (ids, target_ws, target_is, target_sents, target_labels)




