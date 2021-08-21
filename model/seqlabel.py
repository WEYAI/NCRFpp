# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-13 11:49:38

from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wordsequence import WordSequence
from .crf import CRF

class SeqLabel(nn.Module):
    def __init__(self, data):
        super(SeqLabel, self).__init__()
        self.use_crf = data.use_crf
        print("build sequence labeling network...")
        print("use_char: ", data.use_char)
        if data.use_char:
            print("char feature extractor: ", data.char_feature_extractor)
        print("word feature extractor: ", data.word_feature_extractor)
        print("use crf: ", self.use_crf)

        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        ## add two more label for downlayer lstm, use original label size for CRF
        ner_label_size = data.ner_label_alphabet_size
        pos_label_size = data.pos_label_alphabet_size
        chunk_label_size = data.chunk_label_alphabet_size
        data.ner_label_alphabet_size += 2
        data.pos_label_alphabet_size += 2
        data.chunk_label_alphabet_size += 2
        self.word_hidden = WordSequence(data)
        if self.use_crf:
            self.ner_crf = CRF(ner_label_size, self.gpu) # 10  8+2
            self.pos_crf = CRF(pos_label_size, self.gpu) # 46  44+2
            self.chunk_crf = CRF(chunk_label_size, self.gpu) # 22 20+2 


    def calculate_loss(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask):
        outs = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        ner_outs = outs['ner_outputs']
        pos_outs = outs['pos_outputs']
        chunk_outs = outs['chunk_outputs']
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        ner_batch_label = batch_label['ner_batch_label']
        pos_batch_label = batch_label['pos_batch_label']
        chunk_batch_label = batch_label['chunk_batch_label']
        if self.use_crf:
            ner_loss = self.ner_crf.neg_log_likelihood_loss(ner_outs, mask, ner_batch_label)
            pos_loss = self.pos_crf.neg_log_likelihood_loss(pos_outs, mask, pos_batch_label)
            chunk_loss = self.chunk_crf.neg_log_likelihood_loss(chunk_outs, mask, chunk_batch_label)
            ner_scores, ner_tag_seq = self.ner_crf._viterbi_decode(ner_outs, mask)
            pos_scores, pos_tag_seq = self.pos_crf._viterbi_decode(pos_outs, mask)
            chunk_scores, chunk_tag_seq = self.chunk_crf._viterbi_decode(chunk_outs, mask)
            total_loss = ner_loss + pos_loss + chunk_loss
            tag_seq = {
                "ner_batch_seq":ner_tag_seq,
                "pos_batch_seq":pos_tag_seq,
                "chunk_batch_seq":chunk_tag_seq,
            }
        
        else:
            loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
            outs = outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            total_loss = loss_function(score, batch_label.view(batch_size * seq_len))
            _, tag_seq  = torch.max(score, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
        if self.average_batch:
            total_loss = total_loss / batch_size
            tag_seq = {
                "ner_batch_seq":ner_tag_seq,
                "pos_batch_seq":pos_tag_seq,
                "chunk_batch_seq":chunk_tag_seq,
            }
        return total_loss, tag_seq


    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
        outs = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        ner_outs = outs["ner_outputs"] 
        pos_outs = outs["pos_outputs"] 
        chunk_outs = outs["chunk_outputs"] 
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        if self.use_crf:
            ner_scores, ner_tag_seq = self.ner_crf._viterbi_decode(ner_outs, mask)
            pos_scores, pos_tag_seq = self.pos_crf._viterbi_decode(pos_outs, mask)
            chunk_scores, chunk_tag_seq = self.chunk_crf._viterbi_decode(chunk_outs, mask)
            tag_seq = {
                "ner_tag_seq":ner_tag_seq,
                "pos_tag_seq":pos_tag_seq,
                "chunk_tag_seq":chunk_tag_seq,
            }
        else:
            outs = outs.view(batch_size * seq_len, -1)
            _, tag_seq  = torch.max(outs, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            ## filter padded position with zero
            tag_seq = mask.long() * tag_seq
        return tag_seq


    # def get_lstm_features(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
    #     return self.word_hidden(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)



    def decode_nbest(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, nbest):
        if not self.use_crf:
            print("Nbest output is currently supported only for CRF! Exit...")
            exit(0)
        outs = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        scores, tag_seq = self.crf._viterbi_decode_nbest(outs, mask, nbest)
        return scores, tag_seq

