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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning with LM-LSTM-CRF together with Language Model')
    # files
    parser.add_argument('--train_file', nargs='+', default='./corpus/BC5CDR-IOBES/train.tsv', help='path to training file')
    parser.add_argument('--dev_file', nargs='+', default='./corpus/BC5CDR-IOBES/devel.tsv', help='path to development file')
    parser.add_argument('--test_file', nargs='+', default='./corpus/BC5CDR-IOBES/test.tsv', help='path to test file')
    parser.add_argument('--checkpoint', default='./checkpoint/', help='checkpoint path')

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
    parser.add_argument('--idea', default=None)
    parser.add_argument('--pred_method', default='')
    parser.add_argument('--sigmoid', default='')
    parser.add_argument('--mask_value', type=float, default=None)
    
    
    args = parser.parse_args()
    
    assert args.pred_method in ['U', 'M']
    assert args.sigmoid in ['nosig', 'relu']
    assert args.mask_value is not None
    assert args.idea
    
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

    print("can_load_check_point: ", can_load_check_point)
    print("can_load_arg: ", can_load_arg)
    print("load_opt: ", args.load_opt)
    
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
        pickle.dump(trainsets, open( args.pickle + "/temp_trainsets.p", "wb" ))

    print("Up round by batch size")
    print("combined train[idx/len]: ", {crf_idx: sum(map(lambda t: len(t), dataloader)) * args.batch_size  for crf_idx, dataloader in crf2train_dataloader.items()})

    try:
        print("Load from PICKLE")
        devset = pickle.load(open(args.pickle + "/temp_devsets.p", "rb" ))
        crf2dev_dataloader = {}
        for crf_idx, datasets_tuple in devset:
            crf2dev_dataloader[crf_idx] = [torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in datasets_tuple]
    except:
        print("Rebuild")
        crf2dev_dataloader = build_crf2dataloader(crf2corpus, dev_features, dev_labels, 50, corpus_missing_tagspace, args.corpus_mask_value, tag2idx, chr2idx, token2idx, args.caseless, shuffle=False, drop_last=False)
        print("DUMP temp_devsets")
        devsets = [] 
        for crf_idx, dataloaders in crf2dev_dataloader.items():
            devsets.append((crf_idx, [dl.dataset for dl in dataloaders]))
        pickle.dump(devsets, open( args.pickle + "/temp_devsets.p", "wb" ))
    print("combined dev[idx/len]: ", {crf_idx: sum(map(lambda t: len(t), devloader)) * 50 for crf_idx, devloader in crf2dev_dataloader.items()}) 
    

    try:
        print("Load from PICKLE")
        single_devset = pickle.load(open(args.pickle + "/temp_single_dev.p", "rb" ))
        dev_dataset_loader = []
        for datasets_tuple in single_devset:
            dev_dataset_loader.append([torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in datasets_tuple])
    except:
        print("Rebuild")
        dev_dataset_loader = []
        for i in range(num_corpus):
            # construct dataset
            dev_missing_tagspace = [corpus_missing_tagspace[i]] * len(dev_labels[i])
            dev_dataset_loader.append(build_dataloader(dev_features[i], dev_labels[i], 50, dev_missing_tagspace, args.corpus_mask_value, tag2idx, chr2idx, token2idx, args.caseless, shuffle=False, drop_last=False))
        print("DUMP temp_single_dev")
        single_dev_datasets = []
        for dataloader in dev_dataset_loader:
            single_dev_datasets.append([dl.dataset for dl in dataloader])
        pickle.dump(single_dev_datasets, open( args.pickle + "/temp_single_dev.p", "wb" ))
    
    for item in itertools.chain.from_iterable(dev_dataset_loader[0]):
        print("peek cropus mask")
        print(item[8][0])
        break
    
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

    if args.update == 'sgd':
        optimizer = optim.SGD(filter(lambda param: param.requires_grad, ner_model.parameters()), lr=args.lr, momentum=args.momentum)
    elif args.update == 'adam':
        optimizer = optim.Adam(filter(lambda param: param.requires_grad, ner_model.parameters()), lr=args.lr)

    if can_load_check_point and args.load_opt:
        optimizer.load_state_dict(checkpoint_file['optimizer'])

    crit_lm = nn.CrossEntropyLoss()
    if args.max_margin:
        print("Objective Function: Max Margin")
        crit_ner = CRFLoss_mm(len(tag2idx), tag2idx['<start>'], tag2idx['<pad>'], tag2idx['O'], cost_value=args.cost_value, change_gold=args.change_gold, change_prob=args.change_prob)
    elif args.softmax_margin:
        print("Objective Function: Modified Max Margin")
        crit_ner = CRFLoss_sf(tag2idx, tag2idx['<start>'], tag2idx['<pad>'], tag2idx['O'], cost_value=args.cost_value, change_gold=args.change_gold, change_prob=args.change_prob)
    elif args.nll_rp:
        print("Objective Function: Negative Log Liklihood with RP")
        crit_ner = CRFLoss_rp(tag2idx, tag2idx['<start>'], tag2idx['<pad>'], tag2idx['O'], change_prob=args.change_prob)
    else:
        print("Objective Function: Negative Log Liklihood")
        crit_ner = CRFLoss_vb(len(tag2idx), tag2idx['<start>'], tag2idx['<pad>'], O_idx=tag2idx['O'])
    
    if args.gpu >= 0:
        crit_ner.cuda()
        crit_lm.cuda()
        ner_model.cuda()
        packer = CRFRepack_WC(len(tag2idx), True)
    else:
        packer = CRFRepack_WC(len(tag2idx), False)

    if args.start_epoch != 0:
        args.start_epoch += 1
        args.epoch = args.start_epoch + args.epoch
        epoch_list = range(args.start_epoch, args.epoch)
    else:
        args.epoch += 1
        epoch_list = range(1, args.epoch)

    predictor = Predictor(tag2idx, packer, label_seq = True, batch_size = 50)
    evaluator = Evaluator(predictor, packer, tag2idx, args.eva_matrix, args.pred_method)
    
    trainer = Trainer(ner_model, packer, crit_ner, crit_lm, optimizer, evaluator, crf2corpus, args.plateau)
    trainer.train(crf2train_dataloader, crf2dev_dataloader, dev_dataset_loader, epoch_list, args)
    
    trainer.eval_batch_corpus(dev_dataset_loader, args.dev_file, args.corpus2crf)
    
    try:
        print("Load from PICKLE")
        single_testset = pickle.load(open(args.pickle + "/temp_single_test.p", "rb" ))
        test_dataset_loader = []
        for datasets_tuple in single_testset:
            test_dataset_loader.append([torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in datasets_tuple])
    except:
        print("Rebuild")
        test_dataset_loader = []
        for i in range(num_corpus):
            test_missing_tagspace = [corpus_missing_tagspace[i]] * len(test_labels[i])
            test_dataset_loader.append(build_dataloader(test_features[i], test_labels[i], 50, test_missing_tagspace, args.corpus_mask_value, tag2idx, chr2idx, token2idx, args.caseless, shuffle=False, drop_last=False))
        single_test_datasets = []
        for dataloader in test_dataset_loader:
            single_test_datasets.append([dl.dataset for dl in dataloader])
        print("DUMP temp_single_test")
        pickle.dump(single_test_datasets, open( args.pickle + "/temp_single_test.p", "wb" ))

    trainer.eval_batch_corpus(test_dataset_loader, args.test_file, args.corpus2crf)
