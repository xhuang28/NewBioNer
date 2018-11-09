from __future__ import print_function
import datetime
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
from model.crf import *
from model.lm_lstm_crf import *
import model.utils as utils
from model.evaluator import eval_wc, Evaluator
from model.predictor import Predictor #NEW

import argparse
import json
import os
import sys
from tqdm import tqdm
import itertools
import functools
import random
from collections import Counter

from model.data_util import *
from model.trainer import Trainer
import pickle
from sklearn.preprocessing import normalize
import upredict


def tg2target(tg):
    target = (tg / 35).view(tg.shape[0], tg.shape[1])
    return target

def target2tg(target, pad_idx, start_idx, add_first_line=False):
    if add_first_line:
        target = autograd.Variable(torch.LongTensor(np.array([[start_idx for r in range(target.shape[1])]] + target.cpu().data.numpy().tolist()))).cuda()
    target_next = autograd.Variable(torch.LongTensor(target.cpu().data.numpy().tolist()[1:] + [[pad_idx for r in range(target.shape[1])]])).cuda()
    tg = target * 35 + target_next
    tg = tg.view(tg.shape[0], tg.shape[1], 1)
    return tg


def unrestricted_forward_algo(scores, target, mask, corpus_mask, sigmoid, self_args):
    # Restricted Forward Algorithm v1
    # "O": Set scores of all local labels (not including "O") to 0
    # "NE": Set scores of all other labels to 0
    partitions = []
    seq_len = scores.size(0)
    bat_size = scores.size(1)
    
    seq_iter = enumerate(scores)
    # the first score should start with <start>
    _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
    # only need start from start_tag
    cur_partition = inivalues[:, self_args['start_tag'], :]  # bat_size * to_target_size
    partitions.append(cur_partition)
    # iter over last scores
    for idx, cur_values in seq_iter:
        # previous to_target is current from_target
        # cur_partition: previous->current results log(exp(from_target)), #(batch_size * from_target)
        # cur_values: bat_size * from_target * to_target
        cur_values = cur_values + cur_partition.contiguous().view(bat_size, self_args['tagset_size'], 1).expand(bat_size, self_args['tagset_size'], self_args['tagset_size'])
        cur_partition = utils.log_sum_exp(cur_values, self_args['tagset_size'])
        
        partitions.append(cur_partition)
    
    return partitions

def unrestricted_backward_algo(scores, target, mask, corpus_mask, sigmoid, self_args):
    # Restricted Forward Algorithm v1
    # "O": Set scores of all local labels (not including "O") to 0
    # "NE": Set scores of all other labels to 0
    
    rev_scores = autograd.Variable(torch.FloatTensor(np.flip(scores.cpu().data.numpy(), axis=0).copy()).cuda())
    rev_scores = torch.transpose(rev_scores, 2, 3)
    
    rev_target = np.array([[-1 for r in range(target.shape[1])]] + np.flip(target.cpu().data.numpy(), axis=0).tolist()[:-1])
    rev_target = autograd.Variable(torch.LongTensor(rev_target).cuda())
        
    partitions = []
    seq_len = rev_scores.size(0)
    bat_size = rev_scores.size(1)
    
    seq_iter = enumerate(rev_scores)
    _, inivalues = seq_iter.__next__()  # bat_size * to_target_size * from_target_size
    cur_partition = inivalues[:, self_args['end_tag'], :]  # bat_size * from_target_size
    partitions = [cur_partition] + partitions
    # iter over last scores
    for idx, cur_values in seq_iter:
        # previous to_target is current from_target
        # cur_partition: previous->current results log(exp(from_target)), #(batch_size * from_target)
        # cur_values: bat_size * from_target * to_target
        cur_values = cur_values + cur_partition.contiguous().view(bat_size, self_args['tagset_size'], 1).expand(bat_size, self_args['tagset_size'], self_args['tagset_size'])
        cur_partition = utils.log_sum_exp(cur_values, self_args['tagset_size'])
        
        partitions = [cur_partition] + partitions
    
    return partitions

def unrestricted_viterbi_decode(scores, target, mask, corpus_mask, sigmoid, self_args):
    # Restricted Forward Algorithm v1
    # "O": Set scores of all local labels (not including "O") to 0
    # "NE": Set scores of all other labels to 0
    seq_len = scores.size(0)
    bat_size = scores.size(1)
    mask = autograd.Variable(1 - mask.data)
    #decode_idx = scores.new(seq_len-1, bat_size).long()
    decode_idx = torch.LongTensor(seq_len-1, bat_size)
    seq_iter = enumerate(scores)
    # the first score should start with <start>
    _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
    # only need start from start_tag
    forscores = inivalues[:, self_args['start_tag'], :]  # bat_size * to_target_size
    back_points = list()
    # iter over last scores
    for idx, cur_values in seq_iter:
        cur_values = cur_values + forscores.contiguous().view(bat_size, self_args['tagset_size'], 1).expand(bat_size, self_args['tagset_size'], self_args['tagset_size'])
        forscores, cur_bp = torch.max(cur_values, 1)
        cur_bp.masked_fill_(mask[idx].view(bat_size, 1).expand(bat_size, self_args['tagset_size']), self_args['end_tag'])
        back_points.append(cur_bp.data)
    pointer = back_points[-1][:, self_args['end_tag']]
    decode_idx[-1] = pointer
    for idx in range(len(back_points)-2, -1, -1):
        pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(bat_size, 1))
        decode_idx[idx] = pointer
    return decode_idx


def run_upredict_p3(args):
    
    assert args.sigmoid in ['nosig', 'relu']
    assert args.halfway in [0,1]
    assert args.idx_combination != 0
    
    if not torch.cuda.is_available():
        args.gpu = -1

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        if args.seed >= 0:
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

    #JHU cluster
    a = torch.FloatTensor(1).cuda()
    print("OCCPUIED the GPU")
    
    can_load_check_point = args.load_check_point and os.path.isfile(args.load_check_point)
    can_load_arg = args.load_arg and os.path.isfile(args.load_arg)

    if can_load_check_point and can_load_arg:
        with open(args.load_arg, 'r') as f:
            checkpoint_train_args = json.load(f)
        train_args = checkpoint_train_args['args']
        print("Original GPU: ", train_args["gpu"], "Current GPU: ", args.gpu)
        checkpoint_file = torch.load(args.load_check_point, map_location={'cuda:'+str(train_args["gpu"]):'cuda:'+str(args.gpu)})
    else:
        args.load_opt=False 
    print('setting:')
    print(args)

    assert can_load_check_point
    assert can_load_arg
    
    # load corpus
    print('loading corpus')

    # ensure the format of training samples
    if type(args.train_file) == str:
        args.train_file = [args.train_file]
    if type(args.test_as_train) == str:
        args.test_as_train = [args.test_as_train]
    

    print(args.train_file)
    print(args.test_as_train)    
    
    print("load from checkpoint")
    assert not args.combine
    
    if args.train_file != ["0"]:
        train_features, train_labels = read_data(args.train_file)
    
    args.start_epoch = checkpoint_file['epoch']
    token2idx = train_args['token2idx']
    tag2idx = train_args['tag2idx']
    chr2idx = train_args['chr2idx']
    in_doc_words = train_args['in_doc_words']
    print(tag2idx)

    print('constructing dataset')
    
    if args.train_file != ["0"]:
        corpus_missing_tagspace = build_corpus_missing_tagspace(train_labels, tag2idx)
        print("corpus_missing_tagspace", corpus_missing_tagspace)
    
        corpus2crf, corpus_str2crf = corpus_dispatcher(corpus_missing_tagspace, style=args.dispatch)
        print("corpus2crf", corpus2crf)
        print("corpus_str2crf", corpus_str2crf)

        crf2corpus = {}
        for key, val in corpus2crf.items():
            if val not in crf2corpus:
                crf2corpus[val] = [key]
            else:
                crf2corpus[val] += [key]
        print("crf2corpus", crf2corpus)
    
        print("rebuild")
        crf2train_dataloader = build_crf2dataloader(crf2corpus, train_features, train_labels, args.batch_size, corpus_missing_tagspace, args.corpus_mask_value, tag2idx, chr2idx, token2idx, args.caseless, shuffle=True, drop_last=False) 
    
    else:
        print('no training file')
    
    
    print("rebuild")
    # load test sets as train sets
    test_dataloaders = []
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    for file_name in args.test_as_train:
        pred_lines = []
        with codecs.open(file_name, 'r', 'utf-8') as f:
            pred_lines = f.readlines()
        pred_tokens, gold_labels = utils.read_corpus(pred_lines)

        pred_corpus_tagspace = set(["<start>", "<pad>"])
        for sent_labels in gold_labels:
            pred_corpus_tagspace |= set(sent_labels)
        
        crf2pred_missing_tagspace = {}
        crf2train_corpus_tagspace = {}
        crf2train_corpus_missing_tagspace = {}
        crf2pred_missing_tagspace = {0: set(tag2idx) - pred_corpus_tagspace}
        crf2train_corpus_tagspace = {0: set(tag2idx)}
        crf2train_corpus_missing_tagspace = {0: set()}
        
        for crf_idx, curr_pred_missing_tagspace in crf2pred_missing_tagspace.items():
            pred_unique_tagspace = pred_corpus_tagspace - crf2train_corpus_tagspace[crf_idx]
            filtered_gold_labels = gold_labels
            # filter the gold tags outside the training scope
            if pred_unique_tagspace:
                #print("Tagspace outside training scopre, ", pred_unique_tagspace)
                filtered_gold_labels = [[token_label if token_label not in pred_unique_tagspace else 'O' for token_label in sent_labels] for sent_labels in gold_labels]
                
            for pt, fgl in zip(pred_tokens, filtered_gold_labels):
                assert len(pt) == len(fgl), "inconsisnt sent len after filter\n {}\n {}".format(pt, fgl)
            
            pred_missing_tagspace = sorted([tag2idx[untag] for untag in curr_pred_missing_tagspace]) 
            pred_missing_tagspace = [pred_missing_tagspace] * len(filtered_gold_labels)

            dataloader = build_dataloader(pred_tokens, filtered_gold_labels, 50, pred_missing_tagspace, 0, tag2idx, chr2idx, token2idx, train_args['caseless'], shuffle=False, drop_last=False)
            test_dataloaders.append(dataloader)
        
    
    args.token2idx = token2idx
    args.chr2idx = chr2idx
    args.tag2idx = tag2idx
    args.in_doc_words = in_doc_words


    # build model
    print('building model')

    ner_model = LM_LSTM_CRF(len(tag2idx), len(chr2idx), 
        args.char_dim, args.char_hidden, args.char_layers, 
        args.word_dim, args.word_hidden, args.word_layers, len(token2idx), 
        args.drop_out, 1, 
        large_CRF=args.small_crf, if_highway=args.high_way, 
        in_doc_words=in_doc_words, highway_layers = args.highway_layers, sigmoid = args.sigmoid)
    
    if can_load_check_point:
        ner_model.load_state_dict(checkpoint_file['state_dict'])
    else:
        if not args.rand_embedding:
            ner_model.load_pretrained_word_embedding(embedding_tensor)
        ner_model.rand_init(init_word_embedding=args.rand_embedding)
    ner_model.display()
    
    if args.gpu >= 0:
        ner_model.cuda()
        packer = CRFRepack_WC(len(tag2idx), True)
    else:
        packer = CRFRepack_WC(len(tag2idx), False)
    
    
    def unified_predict(dataloader_iter, flag):
        result_p32, result_p33 = [], []
        idx = 0
        for f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v, corpus_mask_v, reorder in dataloader_iter:
            
            idx += 1
            if idx % 1000 == 0:
                print(idx)
            
            f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, corpus_mask_v = packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v, corpus_mask_v)
            
            ner_model.eval()
            scores = ner_model(f_f, f_p, b_f, b_p, w_f, 0, corpus_mask_v)
            
            self_args = {'start_tag': args.tag2idx['<start>'], 'end_tag': args.tag2idx['<pad>'], 'O_idx': args.tag2idx['O'], 'tagset_size': len(args.tag2idx)}
            
            target = tg2target(tg_v)
            
            if flag == 1:
                silver_p32 = upredict.restricted_viterbi_decode(scores, target, mask_v, corpus_mask_v, args.sigmoid, self_args)
            else:
                silver_p32 = unrestricted_viterbi_decode(scores, target, mask_v, corpus_mask_v, args.sigmoid, self_args)
            silver_p32 = autograd.Variable(silver_p32).cuda()
            silver_p32 = target2tg(silver_p32, args.tag2idx['<pad>'], args.tag2idx['<start>'], True)
            
            result_p32.append([f_f.cpu(), f_p.cpu(), b_f.cpu(), b_p.cpu(), w_f.cpu(), silver_p32.cpu(), mask_v.cpu(), len_v.cpu(), corpus_mask_v.cpu(), reorder.cpu()])
            
            if flag == 1:
                for_partitions = upredict.restricted_forward_algo(scores, target, mask_v, corpus_mask_v, args.sigmoid, self_args)
                back_partitions = upredict.restricted_backward_algo(scores, target, mask_v, corpus_mask_v, args.sigmoid, self_args)
            else:
                for_partitions = unrestricted_forward_algo(scores, target, mask_v, corpus_mask_v, args.sigmoid, self_args)
                back_partitions = unrestricted_backward_algo(scores, target, mask_v, corpus_mask_v, args.sigmoid, self_args)
            
            for_partitions = for_partitions[:-1]
            back_partitions = back_partitions[1:]
            
            for_back_partitions = [for_partitions[i] + back_partitions[i] for i in range(len(for_partitions))]
            
            for_back_partitions = [r.cpu().data.numpy().tolist() for r in for_back_partitions]
            
            norm_forback_partitions = []
            
            for i in for_back_partitions:
                new_i = []
                for j in i:
                    norm_j = np.array(j) - np.mean(j)
                    norm_j = np.exp(norm_j)
                    norm_j /= np.sum(norm_j)
                    new_i.append(norm_j.tolist())
                norm_forback_partitions.append(new_i)
            
            log_norm_forback_partitions = np.log(norm_forback_partitions)
            
            result_p33.append([f_f.cpu(), f_p.cpu(), b_f.cpu(), b_p.cpu(), w_f.cpu(), [log_norm_forback_partitions, tg_v.cpu()], mask_v.cpu(), len_v.cpu(), corpus_mask_v.cpu(), reorder.cpu()])
        return (result_p32, result_p33)
    
    if args.train_file != ["0"]:
        full_result_p32, full_result_p33 = unified_predict(itertools.chain.from_iterable(crf2train_dataloader[0]), 1)
    else:
        full_result_p32, full_result_p33 = [], []
    print("train_dataloader complete")
    for dl in test_dataloaders:
        result_p32, result_p33 = unified_predict(itertools.chain.from_iterable(dl), 2)
        full_result_p32 += result_p32
        full_result_p33 += result_p33
        print("test_dataloader complete")
    
    return (full_result_p32, full_result_p33)
    