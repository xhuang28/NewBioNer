
from __future__ import print_function

import torch
import codecs
import model.utils as utils
from collections import Counter

def corpus_dispatcher(corpus_tagspace, style="N2N"):
    corpus2crf = {}
    corpus_str2crf = {}
    if style == "N2N":
        for i, ctg in enumerate(corpus_tagspace):
            str_ctg = " ".join(map(str, ctg))
            if str_ctg not in corpus_str2crf:
                corpus_str2crf[str_ctg] = [i]
            else:
                corpus_str2crf[str_ctg] += [i]

            corpus2crf[i] = i

    if style == "N21":
        for i, ctg in enumerate(corpus_tagspace):
            str_ctg = " ".join(map(str, ctg))
            corpus_str2crf[str_ctg] = [0]
            corpus2crf[i] = 0

    if style == "N2K": 
        for i, ctg in enumerate(corpus_tagspace):
            str_ctg = " ".join(map(str, ctg))
            if str_ctg not in corpus_str2crf:
                corpus_str2crf[str_ctg] = [len(corpus_str2crf)]
            corpus2crf[i] = corpus_str2crf[str_ctg][0]
    return corpus2crf, corpus_str2crf

def build_corpus_missing_tagspace(labels, tag2idx, start="<start>", pad="<pad>"):
    corpus_missing_tagspace = []
    for corpus_label in labels:
        #corpus_tagset = set([tag2idx["<start>"], tag2idx["<pad>"]])
        corpus_tagset = set(["<start>", "<pad>"])
        for sent_labels in corpus_label:
            #corpus_tagset |= set([tag2idx[label] for label in sent_labels])
            corpus_tagset |= set(sent_labels)
            
        unappeared_tags = [tag2idx[untag] for untag in set(tag2idx.keys()) - corpus_tagset]
        corpus_missing_tagspace.append(sorted(unappeared_tags))
    return corpus_missing_tagspace

def build_dataloader(token_features, labels, batch_size, missing_tagspace, corpus_mask_value, tag2idx, chr2idx, token2idx, caseless, shuffle=False, drop_last=False):
    dataset, forw_dev, back_dev = utils.construct_bucket_mean_vb_wc(token_features, labels, missing_tagspace, tag2idx, chr2idx, token2idx, caseless, corpus_mask_value)
    dataloader = [torch.utils.data.DataLoader(tup, batch_size, shuffle=shuffle, drop_last=drop_last) for tup in dataset]
    return dataloader

def build_crf2dataloader(crf2corpus, features, labels, batch_size, corpus_missing_tagspace, corpus_mask_value, tag2idx, chr2idx, token2idx, caseless, shuffle=False, drop_last=False):    
    crf2dataloader = {}

    for crf_idx, corpus_idxs in crf2corpus.items():
        combined_feats = []
        combined_labels = []
        combined_missing_tagspace = []

        for cidx in corpus_idxs:
            combined_feats += features[cidx]
            combined_labels += labels[cidx]
            combined_missing_tagspace += [corpus_missing_tagspace[cidx]] * len(features[cidx])

        crf2dataloader[crf_idx] = build_dataloader(combined_feats, combined_labels, batch_size, combined_missing_tagspace, corpus_mask_value, tag2idx, chr2idx, token2idx, caseless, shuffle=shuffle, drop_last=drop_last)
    return crf2dataloader

def read_data(corpus_files, rebuild_maps=False, mini_count=0):
    corpus_data = []
    for corpus_f in corpus_files:
        with codecs.open(corpus_f, 'r', 'utf-8') as f:
            curr_data = f.readlines()
        corpus_data.append(curr_data)

    tokens = []
    labels = []
    
    token2idx = dict()
    tag2idx = dict()
    chr_cnt = dict()
    chr2idx = dict()
    
    for data in corpus_data:
        if rebuild_maps: 
            print('constructing coding table')
            # here token2idx, tag2idx and chr_cnt are doing argmentaion
            curr_tokens, curr_labels, token2idx, tag2idx, chr_cnt = utils.generate_corpus_char(data, token2idx, tag2idx, chr_cnt, c_thresholds=mini_count, if_shrink_w_feature=False)
        else:
            curr_tokens, curr_labels = utils.read_corpus(data)              
        tokens.append(curr_tokens)
        labels.append(curr_labels)

    shrink_char_count = [k for (k, v) in iter(chr_cnt.items()) if v >= mini_count]
    chr2idx = {shrink_char_count[ind]: ind for ind in range(0, len(shrink_char_count))}

    chr2idx['<u>'] = len(chr2idx)  # unk for char
    chr2idx[' '] = len(chr2idx)  # concat for char
    chr2idx['\n'] = len(chr2idx)  # eof for char


    if rebuild_maps:
        return tokens, labels, token2idx, tag2idx, chr2idx
    else:
        return tokens, labels

def read_combine_data(corpus_files, dev_files, rebuild_maps=False, mini_count=0):
    assert len(corpus_files) == len(dev_files)
    corpus_data = []
    for i, corpus_f in enumerate(corpus_files):
        curr_data = []
        with codecs.open(corpus_f, 'r', 'utf-8') as f:
            curr_data += f.readlines()
        curr_data += ['\n']
        with codecs.open(dev_files[i], 'r', 'utf-8') as df:
            curr_data += df.readlines()
        corpus_data.append(curr_data)

    tokens = []
    labels = []
    
    token2idx = dict()
    tag2idx = dict()
    chr_cnt = dict()
    chr2idx = dict()
    
    for data in corpus_data:
        if rebuild_maps: 
            print('constructing coding table')
            # here token2idx, tag2idx and chr_cnt are doing argmentaion
            curr_tokens, curr_labels, token2idx, tag2idx, chr_cnt = utils.generate_corpus_char(data, token2idx, tag2idx, chr_cnt, c_thresholds=mini_count, if_shrink_w_feature=False)
        else:
            curr_tokens, curr_labels = utils.read_corpus(data)              
        tokens.append(curr_tokens)
        labels.append(curr_labels)

    shrink_char_count = [k for (k, v) in iter(chr_cnt.items()) if v >= mini_count]
    chr2idx = {shrink_char_count[ind]: ind for ind in range(0, len(shrink_char_count))}

    chr2idx['<u>'] = len(chr2idx)  # unk for char
    chr2idx[' '] = len(chr2idx)  # concat for char
    chr2idx['\n'] = len(chr2idx)  # eof for char


    if rebuild_maps:
        return tokens, labels, token2idx, tag2idx, chr2idx
    else:
        return tokens, labels
