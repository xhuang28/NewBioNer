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


def restricted_forward_algo(scores, target, mask, corpus_mask, sigmoid, self_args):
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
        curr_labels = target[idx,:]
        
        # mask cur_partition and cur_values to rule out undesired tag sequences
        partition_mask = np.ones(cur_partition.shape)
        values_mask = np.ones(cur_values.shape)
        for i in range(partition_mask.shape[0]):
            curr_label = curr_labels[i].cpu().data.numpy()[0]
            if curr_label == self_args['O_idx']:
                idx_annotated = np.where(corpus_mask[0,i,0].data)[0]
                idx_annotated = np.array([r for r in idx_annotated if r!=self_args['O_idx']]) # exclude "O"
                partition_mask[i,idx_annotated] = 0
                values_mask[i,idx_annotated,:] = 0
            else:
                partition_mask[i,:] = 0
                partition_mask[i,curr_label] = 1
                values_mask[i,:,:] = 0
                values_mask[i,curr_label,:] = 1
        
        partition_mask = autograd.Variable(torch.FloatTensor(partition_mask)).cuda()
        values_mask = autograd.Variable(torch.FloatTensor(values_mask)).cuda()
        if sigmoid == "relu":
            cur_partition = cur_partition * partition_mask
            cur_values = cur_values * values_mask
        else:
            neg_inf_partition = autograd.Variable(torch.FloatTensor(np.full(cur_partition.shape, -1e9))).cuda()
            neg_inf_values = autograd.Variable(torch.FloatTensor(np.full(cur_values.shape, -1e9))).cuda()
            cur_partition = utils.switch(neg_inf_partition, cur_partition.contiguous(), partition_mask).view(cur_partition.shape)
            cur_values = utils.switch(neg_inf_values, cur_values.contiguous(), values_mask).view(cur_values.shape)
        
        cur_values = cur_values + cur_partition.contiguous().view(bat_size, self_args['tagset_size'], 1).expand(bat_size, self_args['tagset_size'], self_args['tagset_size'])
        cur_partition = utils.log_sum_exp(cur_values, self_args['tagset_size'])
        
        partitions.append(cur_partition)
    
    return partitions

def restricted_backward_algo(scores, target, mask, corpus_mask, sigmoid, self_args):
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
        curr_labels = rev_target[idx,:]
        
        # mask cur_partition and cur_values to rule out undesired tag sequences
        partition_mask = np.ones(cur_partition.shape)
        values_mask = np.ones(cur_values.shape)
        for i in range(partition_mask.shape[0]):
            curr_label = curr_labels[i].cpu().data.numpy()[0]
            if curr_label == self_args['O_idx']:
                idx_annotated = np.where(corpus_mask[0,i,0].data)[0]
                idx_annotated = np.array([r for r in idx_annotated if r!=self_args['O_idx']]) # exclude "O"
                partition_mask[i,idx_annotated] = 0
                values_mask[i,idx_annotated,:] = 0
            else:
                partition_mask[i,:] = 0
                partition_mask[i,curr_label] = 1
                values_mask[i,:,:] = 0
                values_mask[i,curr_label,:] = 1
        
        partition_mask = autograd.Variable(torch.FloatTensor(partition_mask)).cuda()
        values_mask = autograd.Variable(torch.FloatTensor(values_mask)).cuda()
        if sigmoid == "relu":
            cur_partition = cur_partition * partition_mask
            cur_values = cur_values * values_mask
        else:
            neg_inf_partition = autograd.Variable(torch.FloatTensor(np.full(cur_partition.shape, -1e9))).cuda()
            neg_inf_values = autograd.Variable(torch.FloatTensor(np.full(cur_values.shape, -1e9))).cuda()
            cur_partition = utils.switch(neg_inf_partition, cur_partition.contiguous(), partition_mask).view(cur_partition.shape)
            cur_values = utils.switch(neg_inf_values, cur_values.contiguous(), values_mask).view(cur_values.shape)
        
        cur_values = cur_values + cur_partition.contiguous().view(bat_size, self_args['tagset_size'], 1).expand(bat_size, self_args['tagset_size'], self_args['tagset_size'])
        cur_partition = utils.log_sum_exp(cur_values, self_args['tagset_size'])
        
        partitions = [cur_partition] + partitions
    
    return partitions

def restricted_viterbi_decode(scores, target, mask, corpus_mask, sigmoid, self_args):
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
        curr_labels = target[idx,:]
        partition_mask = np.ones(forscores.shape)
        values_mask = np.ones(cur_values.shape)
        for i in range(partition_mask.shape[0]):
            curr_label = curr_labels[i].cpu().data.numpy()[0]
            if curr_label == self_args['O_idx']:
                idx_annotated = np.where(corpus_mask[0,i,0].data)[0]
                idx_annotated = np.array([r for r in idx_annotated if r!=self_args['O_idx']]) # exclude "O"
                partition_mask[i,idx_annotated] = 0
                values_mask[i,idx_annotated,:] = 0
            else:
                partition_mask[i,:] = 0
                partition_mask[i,curr_label] = 1
                values_mask[i,:,:] = 0
                values_mask[i,curr_label,:] = 1
        
        partition_mask = autograd.Variable(torch.FloatTensor(partition_mask)).cuda()
        values_mask = autograd.Variable(torch.FloatTensor(values_mask)).cuda()
        if sigmoid == "relu":
            forscores = forscores * partition_mask
            cur_values = cur_values * values_mask
        else:
            neg_inf_partition = autograd.Variable(torch.FloatTensor(np.full(forscores.shape, -1e9))).cuda()
            neg_inf_values = autograd.Variable(torch.FloatTensor(np.full(cur_values.shape, -1e9))).cuda()
            forscores = utils.switch(neg_inf_partition, forscores.contiguous(), partition_mask).view(forscores.shape)
            cur_values = utils.switch(neg_inf_values, cur_values.contiguous(), values_mask).view(cur_values.shape)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning with LM-LSTM-CRF together with Language Model')
    # files
    parser.add_argument('--train_file', nargs='+', default='./corpus/BC5CDR-IOBES/train.tsv', help='path to training file')
    parser.add_argument('--dev_file', nargs='+', default='./corpus/BC5CDR-IOBES/devel.tsv', help='path to development file')
    parser.add_argument('--test_file', nargs='+', default='./corpus/BC5CDR-IOBES/test.tsv', help='path to test file')
    parser.add_argument('--data_loader', default='./data_loader/', help='path to save data_loader')

    # if need to resume training
    parser.add_argument('--load_check_point', default='', help='path previous checkpoint that want to be loaded')
    parser.add_argument('--load_arg', default='', help='path to arg json')
    parser.add_argument('--load_opt', action='store_true', help='also load optimizer from the checkpoint')
    parser.add_argument('--start_epoch', type=int, default=0, help='start point of epoch')
    parser.add_argument('--least_iters', type=int, default=50, help='at least train how many epochs before stop')
    
    #training args
    parser.add_argument('--emb_file', default='./external/embedding/glove.6B.200d.txt', help='path to pre-trained embedding')
    parser.add_argument('--fine_tune', action='store_false', help='fine tune the diction of word embedding or not')
    parser.add_argument('--rand_embedding', action='store_true', help='random initialize word embedding')
    parser.add_argument('--shrink_embedding', action='store_true', help='shrink the embedding dictionary to corpus (open this if pre-trained embedding dictionary is too large, but disable this may yield better results on external corpus)')
    parser.add_argument('--word_dim', type=int, default=200, help='dimension of word embedding')

    parser.add_argument('--unk', default='unk', help='unknow-token in pre-trained embedding')
    parser.add_argument('--caseless', action='store_true', help='caseless or not')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--update', choices=['sgd', 'adam'], default='sgd', help='optimizer choice')
    parser.add_argument('--mini_count', type=float, default=5, help='thresholds to replace rare words with <unk>')
    parser.add_argument('--eva_matrix', choices=['a', 'fa'], default='fa', help='use f1 and accuracy or accuracy alone')
    parser.add_argument('--combine', action='store_true', help='combine training')
    parser.add_argument('--stop_on_single', action='store_true', help='early stop on single corpus')
    parser.add_argument('--plateau', action='store_true', help='adjust learning rate with plateau')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='clip grad at')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    
    #model args
    parser.add_argument('--char_layers', type=int, default=1, help='number of char level layers')
    parser.add_argument('--word_layers', type=int, default=1, help='number of word level layers')
    parser.add_argument('--small_crf', action='store_false', help='use small crf instead of large crf, refer model.crf module for more details')
    parser.add_argument('--co_train', action='store_true', help='cotrain language model')
    parser.add_argument('--lambda0', type=float, default=1, help='lambda0')
    parser.add_argument('--high_way', action='store_true', help='use highway layers')
    parser.add_argument('--highway_layers', type=int, default=1, help='number of highway layers')
    parser.add_argument('--dispatch', choices=['N2N', 'N21', 'N2K'], default="N2N", help='how to combine crf layer')
    
    # tuning params
    parser.add_argument('--seed', type=int, default=42, help='random seed, < 0 not seed, magic number')
    parser.add_argument('--epoch', type=int, default=200, help='maximum epoch number')
    parser.add_argument('--batch_size', type=int, default=10, help='train batch_size')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.05, help='decay ratio of learning rate')
    parser.add_argument('--drop_out', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--patience', type=int, default=30, help='patience for early stop')
    
    parser.add_argument('--char_hidden', type=int, default=200, help='dimension of char-level layers')
    parser.add_argument('--word_hidden', type=int, default=200, help='dimension of word-level layers')
    parser.add_argument('--char_dim', type=int, default=30, help='dimension of char embedding')
    
    parser.add_argument('--corpus_mask_value', type=float, default=0.0, help='corpus_mask_value')
       
    parser.add_argument('--max_margin', action='store_true', help='max_margin loss')
    parser.add_argument('--softmax_margin', action='store_true', help='softmax_margin loss')
    parser.add_argument('--nll_rp', action='store_true', help='nll_rp')
    
    parser.add_argument('--cost_value', type=float, default=1.0, help='argument cost value')
    parser.add_argument('--change_gold', action='store_true', help='change the gold to omit the global prediction')
    parser.add_argument('--change_prob', type=float, default=1.0, help='the prob to perform change gold')

    parser.add_argument('--pickle', default='', help='unknow-token in pre-trained embedding')
    
    # new params
    parser.add_argument('--sigmoid', default='')
    parser.add_argument('--halfway', type=int, default=0)
    
    
    args = parser.parse_args()
    
    assert args.sigmoid in ['nosig', 'relu']
    assert args.halfway in [0,1]
    
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
    if type(args.dev_file) == str:
        args.dev_file = [args.dev_file]
    if type(args.test_file) == str:
        args.test_file = [args.test_file]


    print(args.train_file)
    print(args.dev_file)
    print(args.test_file)


    num_corpus = len(args.train_file)
    
    rebuild_maps = not can_load_check_point or not can_load_arg
    if rebuild_maps:
        print("rebuild_maps")
        if args.combine:
            train_features, train_labels, token2idx, tag2idx, chr2idx = read_combine_data(args.train_file, args.dev_file, rebuild_maps, args.mini_count)
        else:
            train_features, train_labels, token2idx, tag2idx, chr2idx = read_data(args.train_file, rebuild_maps, args.mini_count)
        dev_features, dev_labels = read_data(args.dev_file)
        test_features, test_labels = read_data(args.test_file)

        token_set = {v for v in token2idx}
        dt_token_set = token_set

        train_features_tot = functools.reduce(lambda x, y: x + y, train_features)
        token2idx = utils.shrink_features(token2idx, train_features_tot, args.mini_count)

        for i in range(num_corpus):         
            dt_token_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), dev_features[i]), dt_token_set)
            dt_token_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_features[i]), dt_token_set)

        if not args.rand_embedding:
            print("feature size: '{}'".format(len(token2idx)))
            print('loading embedding')
            if args.fine_tune:  # which means does not do fine-tune
                token2idx = {'<eof>': 0}
            try:
                print("Load from PICKLE")
                token2idx, embedding_tensor, in_doc_words = pickle.load(open(args.pickle + "/temp_pubmed.p", "rb" ))
            except:
                print("Rebuild")
                token2idx, embedding_tensor, in_doc_words = utils.load_embedding_wlm(args.emb_file, ' ', token2idx, dt_token_set, args.caseless, args.unk, args.word_dim, shrink_to_corpus=args.shrink_embedding)
                print("DUMP temp_pubmed.p")
                pickle.dump((token2idx, embedding_tensor, in_doc_words), open( args.pickle + "/temp_pubmed.p", "wb" ))
            print("embedding size: '{}'".format(len(token2idx)))

        tag_set = set()
        for i in range(num_corpus):         
            tag_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), dev_labels[i]), tag_set)
            tag_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_labels[i]), tag_set)

        for label in sorted(tag_set):
            if label not in tag2idx:
                tag2idx[label] = len(tag2idx)
    else:
        print("load from checkpoint")
        if args.combine:
            train_features, train_labels = read_combine_data(args.train_file, args.dev_file)
        else:
            train_features, train_labels = read_data(args.train_file)
        dev_features, dev_labels = read_data(args.dev_file)
        test_features, test_labels = read_data(args.test_file)
        
        args.start_epoch = checkpoint_file['epoch']
        token2idx = train_args['token2idx']
        tag2idx = train_args['tag2idx']
        chr2idx = train_args['chr2idx']
        in_doc_words = train_args['in_doc_words']
    print(tag2idx)
    print("Statistic:")
    for i in range(len(args.train_file)):
        tg_cntr = Counter()
        for sent_tags in train_labels[i]:
           tg_cntr += Counter(sent_tags) 
        print(args.train_file[i], len(train_labels[i]))
        print(tg_cntr)
    print()

    for i in range(len(args.train_file)):
        tg_cntr = Counter()
        for sent_tags in dev_labels[i]:
           tg_cntr += Counter(sent_tags) 
        print(args.dev_file[i], len(dev_labels[i]))
        print(tg_cntr)
    print()

    for i in range(len(args.train_file)):
        tg_cntr = Counter()
        for sent_tags in test_labels[i]:
           tg_cntr += Counter(sent_tags) 
        print(args.test_file[i], len(test_labels[i]))
        print(tg_cntr)
    print()

    print('constructing dataset')
    
    corpus_missing_tagspace = build_corpus_missing_tagspace(train_labels, tag2idx)
    print("corpus_missing_tagspace", corpus_missing_tagspace)

    corpus2crf, corpus_str2crf = corpus_dispatcher(corpus_missing_tagspace, style=args.dispatch)
    print("corpus2crf", corpus2crf)
    print("corpus_str2crf", corpus_str2crf)
    for i, filename in enumerate(args.train_file):
        print(filename, "-> CRF: ", corpus2crf[i])

    crf2corpus = {}
    for key, val in corpus2crf.items():
        if val not in crf2corpus:
            crf2corpus[val] = [key]
        else:
            crf2corpus[val] += [key]
    print("crf2corpus", crf2corpus)

    
    try:
        print("Load from PICKLE")
        trainset = pickle.load(open(args.pickle + "/temp_trainsets.p", "rb" ))
        crf2train_dataloader = {}
        for crf_idx, datasets_tuple in trainset:
            crf2train_dataloader[crf_idx] = [torch.utils.data.DataLoader(tup, args.batch_size, shuffle=True, drop_last=False) for tup in datasets_tuple]
    except:
        print("rebuild")
        crf2train_dataloader = build_crf2dataloader(crf2corpus, train_features, train_labels, args.batch_size, corpus_missing_tagspace, args.corpus_mask_value, tag2idx, chr2idx, token2idx, args.caseless, shuffle=True, drop_last=False) 
        trainsets = [] 
        for crf_idx, dataloaders in crf2train_dataloader.items():
            trainsets.append((crf_idx, [dl.dataset for dl in dataloaders]))
        print("Dump temp_trainsets")
        pickle.dump(trainsets, open(args.pickle + "/temp_trainsets.p", "wb" ))

    print("Up round by batch size")
    print("combined train[idx/len]: ", {crf_idx: sum(map(lambda t: len(t), dataloader)) * args.batch_size  for crf_idx, dataloader in crf2train_dataloader.items()})
    
    
    args.token2idx = token2idx
    args.chr2idx = chr2idx
    args.tag2idx = tag2idx
    args.in_doc_words = in_doc_words
    args.corpus_missing_tagspace = corpus_missing_tagspace
    args.corpus2crf = corpus2crf
    args.corpus_str2crf = corpus_str2crf


    # build model
    print('building model')

    ner_model = LM_LSTM_CRF(len(tag2idx), len(chr2idx), 
        args.char_dim, args.char_hidden, args.char_layers, 
        args.word_dim, args.word_hidden, args.word_layers, len(token2idx), 
        args.drop_out, len(crf2corpus), 
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
    
    suffix = 'combine' if args.combine else 'train'
    prefix = '/halfway' if args.halfway else ''
    f_p22 = open(args.data_loader + prefix + '/P22_' + suffix + '.p', 'wb')
    f_p23 = open(args.data_loader + prefix + '/P23_' + suffix + '.p', 'wb')
    
    
    for f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v, corpus_mask_v, reorder in itertools.chain.from_iterable(crf2train_dataloader[0]):
        
        f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, corpus_mask_v = packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v, corpus_mask_v)
        
        ner_model.eval()
        scores = ner_model(f_f, f_p, b_f, b_p, w_f, 0, corpus_mask_v)
        
        self_args = {'start_tag': args.tag2idx['<start>'], 'end_tag': args.tag2idx['<pad>'], 'O_idx': args.tag2idx['O'], 'tagset_size': len(args.tag2idx)}
        
        target = tg2target(tg_v)
        
        silver_p22 = restricted_viterbi_decode(scores, target, mask_v, corpus_mask_v, args.sigmoid, self_args)
        silver_p22 = autograd.Variable(silver_p22).cuda()
        silver_p22 = target2tg(silver_p22, args.tag2idx['<pad>'], args.tag2idx['<start>'], True)
        
        pickle.dump([f_f.cpu(), f_p.cpu(), b_f.cpu(), b_p.cpu(), w_f.cpu(), silver_p22.cpu(), mask_v.cpu(), len_v.cpu(), corpus_mask_v.cpu(), reorder.cpu()], f_p22, 0)
        
        for_partitions = restricted_forward_algo(scores, target, mask_v, corpus_mask_v, args.sigmoid, self_args)
        
        back_partitions = restricted_backward_algo(scores, target, mask_v, corpus_mask_v, args.sigmoid, self_args)
        
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
        
        pickle.dump([f_f.cpu(), f_p.cpu(), b_f.cpu(), b_p.cpu(), w_f.cpu(), [log_norm_forback_partitions, tg_v.cpu()], mask_v.cpu(), len_v.cpu(), corpus_mask_v.cpu(), reorder.cpu()], f_p23, 0)
        
    f_p22.close()
    f_p23.close()
    