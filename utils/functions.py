# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:23:06
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-14 12:23:52
from __future__ import print_function
from __future__ import absolute_import
import sys
import numpy as np

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance(input_file, word_alphabet, char_alphabet, feature_alphabets, ner_label_alphabet,pos_label_alphabet,chunk_label_alphabet, number_normalized, max_sent_length, sentence_classification=False, split_token='\t', char_padding_size=-1, char_padding_symbol = '</pad>'):
    feature_num = len(feature_alphabets)
    in_lines = open(input_file,'r', encoding="utf8").readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    features = []
    chars = []
    ner_labels = []
    pos_labels = []
    chunk_labels = []
    word_Ids = []
    feature_Ids = []
    char_Ids = []
    ner_label_Ids = []
    pos_label_Ids = []
    chunk_label_Ids = []
    ### for sequence labeling data format i.e. CoNLL 2003
    for line in in_lines:
            if len(line) > 2:
                
                pairs = line.strip().split()
                word = pairs[0]
                if word == '-DOCSTART-': 
                    # print("clear dirty data") # clear dirty data
                    continue
                if sys.version_info[0] < 3:
                    word = word.decode('utf-8')
                words.append(word)
                if number_normalized:
                    word = normalize_word(word)
                ner_label = pairs[3]
                pos_label = pairs[1]
                chunk_label = pairs[2]
                ner_labels.append(ner_label)
                pos_labels.append(pos_label)
                chunk_labels.append(chunk_label)
                word_Ids.append(word_alphabet.get_index(word))
                ner_label_Ids.append(ner_label_alphabet.get_index(ner_label))
                pos_label_Ids.append(pos_label_alphabet.get_index(pos_label))
                chunk_label_Ids.append(chunk_label_alphabet.get_index(chunk_label))
                ## get features
                feat_list = []
                feat_Id = []
                for idx in range(feature_num):
                    feat_idx = pairs[idx+1].split(']',1)[-1]
                    feat_list.append(feat_idx)
                    feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
                features.append(feat_list)
                feature_Ids.append(feat_Id)
                ## get char
                char_list = []
                char_Id = []
                for char in word:
                    char_list.append(char)
                if char_padding_size > 0:
                    char_number = len(char_list)
                    if char_number < char_padding_size:
                        char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                    assert(len(char_list) == char_padding_size)
                else:
                    ### not padding
                    pass
                for char in char_list:
                    char_Id.append(char_alphabet.get_index(char))
                chars.append(char_list)
                char_Ids.append(char_Id)
            else:
                if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
                    instence_texts.append([words, features, chars, ner_labels,pos_labels,chunk_labels])
                    instence_Ids.append([word_Ids, feature_Ids, char_Ids,ner_label_Ids,pos_label_Ids,chunk_label_Ids])
                words = []
                features = []
                chars = []
                ner_labels = []
                pos_labels = []
                chunk_labels = []
                 
                word_Ids = []
                feature_Ids = []
                char_Ids = []
                ner_label_Ids = []
                pos_label_Ids = []
                chunk_label_Ids = []
    if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
                instence_texts.append([words, features, chars, ner_labels,pos_labels,chunk_labels])
                instence_Ids.append([word_Ids, feature_Ids, char_Ids,ner_label_Ids,pos_label_Ids,chunk_label_Ids])
                words = []
                features = []
                chars = []
                ner_labels = []
                pos_labels = []
                chunk_labels = []
                word_Ids = []
                feature_Ids = []
                char_Ids = []
                ner_label_Ids = []
                pos_label_Ids = []
                chunk_label_Ids = []
    return instence_texts, instence_Ids
    pass


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim

def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            elif embedd_dim + 1 != len(tokens):
                ## ignore illegal embedding line
                continue
                # assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            if sys.version_info[0] < 3:
                first_col = tokens[0].decode('utf-8')
            else:
                first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim

if __name__ == '__main__':
    a = np.arange(9.0)
    print(a)
    print(norm2one(a))
    
