
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

import argparse
import json
import os
import sys
from tqdm import tqdm
import itertools
import functools
import random
from model.trainer import Trainer

from model.data_util import *

from model.predictor import Predictor
from collections import Counter
import pickle

def global_predict(separate_crf_result):
    # separate_crf_result: list of sents [sent[tokens]]
    global_pred_tags = []
    assert len(set([len(item) for item in separate_crf_result])) == 1, "inconsist pred result {}".format([len(item) for item in separate_crf_result])
    collison = []
    for sent_idx, crf_sent_tags in enumerate(zip(*separate_crf_result)):
        global_sent_tags = []
        collide_idx = []
        for i, crf_token_tag in enumerate(zip(*crf_sent_tags)):
            tag_cntr = Counter(crf_token_tag)
            # entity tags have high priority than O
            if not (set(tag_cntr.keys()) - set(["O"])):
                global_sent_tags.append("O")
            else:
                del tag_cntr["O"]
                global_sent_tags.append(tag_cntr.most_common(1)[0][0])            
                if len(tag_cntr) > 1:
                    collide_idx.append(i)
        # group the collision in sent level
        if collide_idx:
            collison.append((sent_idx, collide_idx, crf_sent_tags))
        global_pred_tags.append(global_sent_tags)
    for g, l in zip(global_pred_tags, separate_crf_result[0]):
        assert len(g) == len(l), "inconsisnt sent len\n {}\n {}".format(g, l)
    return global_pred_tags, collison

def display_collison(collison, tokens, gold_labels):
    collide_tokens = 0
    print("Only consider tag collison, tag collide with O is exclude")
    print("-DOCSTART-")
    print("-SENT_ID- -COLLISION_ID-")
    print("-TOKEN_ID- -TOKEN- -GOLD- -TAGS- -TAGS- -TAGS- ...")
    for sent_idx, collide_idx, crf_pred_sent in collison:
        print(sent_idx, collide_idx)
        collide_tokens += len(collide_idx)
        for i, token in enumerate(zip(tokens[sent_idx], gold_labels[sent_idx], *crf_pred_sent)):
            print("\t".join((str(i), ) + tuple(map(lambda x: x.ljust(20), token))))
        print()
    print("Total Sent: {}, Collide Sents: {}".format(len(tokens), len(collison)))
    print("Total Tokens: {}, Collide Tokens: {}".format(sum([len(item) for item in tokens]), collide_tokens))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluating LM-BLSTM-CRF')
    parser.add_argument('--load_arg', default='./checkpoint/cwlm_lstm_crf.json', help='path to arg json')
    parser.add_argument('--load_check_point', default='./checkpoint/cwlm_lstm_crf.model', help='path to model checkpoint file')
    parser.add_argument('--gpu',type=int, default=0, help='gpu id')
    parser.add_argument('--eva_matrix', choices=['a', 'fa'], default='fa', help='use f1 and accuracy or f1 alone')
    parser.add_argument('--corpus_mask_value', type=float, default=0.0, help='mask to control the prediction scope')
    parser.add_argument('--local_eval', action='store_true', help='whether perform eval on dev and test set of train corpus')
    parser.add_argument('--if_pred', action='store_true', help='whether perform golbal prediction')
    parser.add_argument('--global_eval', action='store_true', help='global evaluation')
    parser.add_argument('--global_pred', action='store_true', help='global prediction')
    parser.add_argument('--pred_scope', choices=['local', 'global'], default='local', help='control the prediction scope, local pred or global')
    parser.add_argument('--pred_file', nargs='+', default='./corpus/BC5CDR-IOBES/test.tsv', help='path to predict file')
    parser.add_argument('--show_collison', action='store_true', help='show collison sent')
    parser.add_argument('--debug_errors', action='store_true', help='debug_errors')
    parser.add_argument('--error_num', type=int, default=100, help='debug_errors')
    parser.add_argument('--verbose',type=int, default=0, help='gpu id')
    parser.add_argument('--post',action='store_true', help='gpu id')
    parser.add_argument('--eval_filtered',action='store_true', help='gpu id')
    parser.add_argument('--pickle', default='', help='unknow-token in pre-trained embedding')
    parser.add_argument('--annotate', action='store_true', help='unknow-token in pre-trained embedding')
    
    # new params
    parser.add_argument('--pred_method', default="")
    

    # parse the eval settings
    args = parser.parse_args()
    print('eval setting:')
    print(args)
    
    assert args.pred_method
    
    # load training settings
    with open(args.load_arg, 'r') as f:
        train_args = json.load(f)
    train_args = train_args['args']
    print('train setting:')
    if "max_margin" in train_args and train_args["max_margin"]:
        print("Max Margin")
    else:
        print("Negative Log Likelihood")

    if not torch.cuda.is_available():
        args.gpu = -1

    #torch.manual_seed(train_args['seed'])
    #random.seed(train_args['seed'])
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        #torch.cuda.manual_seed(train_args['seed'])
        #torch.cuda.manual_seed_all(train_args['seed'])

    if type(args.pred_file) == str:
        args.pred_file = [args.pred_file]

    # load map2idx for construct the prediction and evaluation corpus in the same way as the training 
    if args.gpu >= 0:
        checkpoint_file = torch.load(args.load_check_point, map_location={'cuda:'+str(train_args["gpu"]):'cuda:'+str(args.gpu)})
    else:
        checkpoint_file = torch.load(args.load_check_point, map_location=lambda storage, loc: storage)
    token2idx = train_args['token2idx']
    tag2idx = train_args['tag2idx']
    chr2idx = train_args['chr2idx']
    
    in_doc_words = train_args['in_doc_words']
    corpus2crf = train_args['corpus2crf']
    corpus2crf = {int(key): int(val) for key, val in corpus2crf.items()}
    print("corpus2crf", corpus2crf)
    corpus_str2crf = train_args['corpus_str2crf']
    print("corpus_str2crf", corpus_str2crf)

    crf2corpus = {}
    for key, val in corpus2crf.items():
        if val not in crf2corpus:
            crf2corpus[val] = [key]
        else:
            crf2corpus[val] += [key]
    print("crf2corpus", crf2corpus)

    corpus_missing_tagspace = train_args['corpus_missing_tagspace']
    print("corpus_missing_tagspace", corpus_missing_tagspace)

    # build model
    ner_model = LM_LSTM_CRF(len(tag2idx), len(chr2idx), 
        train_args['char_dim'], train_args['char_hidden'], train_args['char_layers'], 
        train_args['word_dim'], train_args['word_hidden'], train_args['word_layers'], len(token2idx), 
        train_args['drop_out'], len(crf2corpus), 
        large_CRF=train_args['small_crf'], if_highway=train_args['high_way'], 
        in_doc_words=in_doc_words, highway_layers = train_args['highway_layers'], sigmoid = train_args['sigmoid'])
    
    ner_model.load_state_dict(checkpoint_file['state_dict'])
    ner_model.display()
    

    if "max_margin" in train_args and train_args["max_margin"]:
        print("Objective Function: Max Margin")
        crit_ner = CRFLoss_mm(len(tag2idx), tag2idx['<start>'], tag2idx['<pad>'], tag2idx['O'], cost_value=train_args["cost_value"], change_gold=train_args["change_gold"], change_prob=train_args["change_prob"])
    else:
        print("Objective Function: Negative Log Liklihood")
        crit_ner = CRFLoss_vb(len(tag2idx), tag2idx['<start>'], tag2idx['<pad>'], O_idx=tag2idx['O'])
    
    crit_lm = nn.CrossEntropyLoss()
    optimizer = None

    if args.gpu >= 0:
        if_cuda = True
        torch.cuda.set_device(args.gpu)
        ner_model.cuda()
        packer = CRFRepack_WC(len(tag2idx), True)
    else:
        if_cuda = False
        packer = CRFRepack_WC(len(tag2idx), False)


    # init the predtor and evaltor
    # predictor 
    predictor = Predictor(tag2idx, packer, label_seq = True, batch_size = 50)
    
    # evaluator       
    evaluator = Evaluator(predictor, packer, tag2idx, args.eva_matrix, args.pred_method)

    agent = Trainer(ner_model, packer, crit_ner, crit_lm, optimizer, evaluator, crf2corpus)
    
    # perform the evalution for dev and test set of training corpus
    if args.local_eval:
        assert len(train_args['dev_file']) == len(train_args['test_file'])
        num_corpus = len(train_args['dev_file'])


        # construct the pred and eval dataloader
        dev_tokens = []
        dev_labels = []

        test_tokens = []
        test_labels = []

        for i in range(num_corpus):
            dev_lines = []
            with codecs.open(train_args['dev_file'][i], 'r', 'utf-8') as f:
                dev_lines = f.readlines()
            dev_features, dev_l = utils.read_corpus(dev_lines)
            dev_tokens.append(dev_features)
            dev_labels.append(dev_l)

            test_lines = []
            with codecs.open(train_args['test_file'][i], 'r', 'utf-8') as f:
                test_lines = f.readlines()
            test_features, test_l = utils.read_corpus(test_lines)
            test_tokens.append(test_features)
            test_labels.append(test_l)

        """
        try:
            print("Load from PICKLE")
            single_devset = pickle.load(open(args.pickle + "/single_dev.p", "rb" ))
            dev_dataset_loader = []
            for datasets_tuple in single_devset:
                dev_dataset_loader.append([torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in datasets_tuple])
        except:
            print("Rebuild")
            dev_dataset_loader = []
            for i in range(num_corpus):
                # construct dataset
                dev_missing_tagspace = [corpus_missing_tagspace[i]] * len(dev_labels[i])
                dev_dataset_loader.append(build_dataloader(dev_tokens[i], dev_labels[i], 50, dev_missing_tagspace, train_args["corpus_mask_value"], tag2idx, chr2idx, token2idx, train_args['caseless'], shuffle=False, drop_last=False))
            
            print("DUMP temp_single_dev")
            single_dev_datasets = []
            for dataloader in dev_dataset_loader:
                single_dev_datasets.append([dl.dataset for dl in dataloader])
            pickle.dump(single_dev_datasets, open( args.pickle + "/temp_single_dev.p", "wb" ))
        
        try:
            print("Load from PICKLE")
            single_testset = pickle.load(open(args.pickle + "/single_test.p", "rb" ))
            test_dataset_loader = []
            for datasets_tuple in single_testset:
                test_dataset_loader.append([torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in datasets_tuple])
        except:
            print("Rebuild")
            test_dataset_loader = []
            for i in range(num_corpus):
                test_missing_tagspace = [corpus_missing_tagspace[i]] * len(test_labels[i])
                test_dataset_loader.append(build_dataloader(test_features[i], test_labels[i], 50, test_missing_tagspace, train_args["corpus_mask_value"], tag2idx, chr2idx, token2idx, train_args['caseless'], shuffle=False, drop_last=False))
            single_test_datasets = []
            for dataloader in test_dataset_loader:
                single_test_datasets.append([dl.dataset for dl in dataloader])
            print("DUMP temp_single_test")
            pickle.dump(single_test_datasets, open( args.pickle + "/temp_single_test.p", "wb" ))
        """
        
        try:
            print("Load from PICKLE")
            single_devset = pickle.load(open(args.pickle + "/temp_single_dev.p", "rb" ))
            dev_dataset_loader = []
            for datasets_tuple in single_devset:
                dev_dataset_loader.append([torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in datasets_tuple])
            
            print("Load from PICKLE")
            single_testset = pickle.load(open(args.pickle + "/temp_single_test.p", "rb" ))
            test_dataset_loader = []
            for datasets_tuple in single_testset:
                test_dataset_loader.append([torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in datasets_tuple])
        except:
            dev_dataset_loader = []
            test_dataset_loader = []
            for i in range(num_corpus):
                # construct dataset
                dev_missing_tagspace = [corpus_missing_tagspace[i]] * len(dev_labels[i])
                test_missing_tagspace = [corpus_missing_tagspace[i]] * len(test_labels[i])

                dev_dataset_loader.append(build_dataloader(dev_tokens[i], dev_labels[i], 50, dev_missing_tagspace, train_args["corpus_mask_value"], tag2idx, chr2idx, token2idx, train_args['caseless'], shuffle=False, drop_last=False))
                test_dataset_loader.append(build_dataloader(test_tokens[i], test_labels[i], 50, test_missing_tagspace, train_args["corpus_mask_value"], tag2idx, chr2idx, token2idx, train_args['caseless'], shuffle=False, drop_last=False))

            single_dev_datasets = []
            for dataloader in dev_dataset_loader:
                single_dev_datasets.append([dl.dataset for dl in dataloader])
            pickle.dump(single_dev_datasets, open( args.pickle + "/temp_single_dev.p", "wb" ))

            single_test_datasets = []
            for dataloader in test_dataset_loader:
                single_test_datasets.append([dl.dataset for dl in dataloader])
            print("DUMP temp_single_test")
            pickle.dump(single_test_datasets, open( args.pickle + "/temp_single_test.p", "wb" ))
        



        #agent.eval_batch_corpus(dev_dataset_loader, train_args['dev_file'], corpus2crf)
        agent.eval_batch_corpus(test_dataset_loader, train_args['test_file'], corpus2crf)

    # global prediction
    if args.if_pred:
        idx2tag = {idx: tag for tag, idx in tag2idx.items()}
        for file_name in args.pred_file:
            pred_lines = []
            with codecs.open(file_name, 'r', 'utf-8') as f:
                pred_lines = f.readlines()
            pred_tokens, gold_labels = utils.read_corpus(pred_lines)

            pred_corpus_tagspace = set(["<start>", "<pad>"])
            for sent_labels in gold_labels:
                pred_corpus_tagspace |= set(sent_labels)

            # local pred: use crfs binds to each training corpus
            # so you need to compute the mask for this training branch to get overlapped tags
            if args.pred_scope == "local":
                crf2pred_missing_tagspace = {}
                crf2train_corpus_tagspace = {}
                crf2train_corpus_missing_tagspace = {}
                # compute the mask for each training brach
                if train_args["dispatch"] in ["N2N", "N2K"]:
                    for corpus_str, crf_idxs in corpus_str2crf.items():
                        for crf_idx in crf_idxs:
                            if corpus_str == "":
                                train_corpus_missing_tagspace = set()
                            else:
                                train_corpus_missing_tagspace = map(int, corpus_str.split(" "))
                                train_corpus_missing_tagspace = set([idx2tag[untag_idx] for untag_idx in train_corpus_missing_tagspace])
                            train_corpus_tagspace = set(tag2idx.keys() - train_corpus_missing_tagspace)
                            curr_pred_missing_tagspace = set(tag2idx) - (pred_corpus_tagspace & train_corpus_tagspace)
                            
                            if crf_idx in crf2train_corpus_tagspace:
                                assert crf2train_corpus_tagspace[crf_idx] == train_corpus_tagspace
                            else:
                                crf2train_corpus_tagspace[crf_idx] = train_corpus_tagspace

                            if crf_idx in crf2train_corpus_missing_tagspace:
                                assert crf2train_corpus_missing_tagspace[crf_idx] == train_corpus_missing_tagspace
                            else:
                                crf2train_corpus_missing_tagspace[crf_idx] = train_corpus_missing_tagspace
                            
                            # the target has nothing in common with the training crf brach, no need to predict with this crf
                            if curr_pred_missing_tagspace  == (set(tag2idx) - set(["<start>", "<pad>", "O"])):
                                print("Nothing same, skip, ", crf_idx, curr_pred_missing_tagspace | set(["<start>", "<pad>", "O"]), set(tag2idx))
                                #continue
                            
                            if crf_idx in crf2pred_missing_tagspace:
                                assert crf2pred_missing_tagspace[crf_idx] == curr_pred_missing_tagspace
                            else:
                                crf2pred_missing_tagspace[crf_idx] = curr_pred_missing_tagspace
                elif train_args["dispatch"] == "N21":
                    crf2pred_missing_tagspace = {0: set(tag2idx) - pred_corpus_tagspace}
                    crf2train_corpus_tagspace = {0: set(tag2idx)}
                    crf2train_corpus_missing_tagspace = {0: set()}

                if args.verbose >= 2:
                    print("pred_corpus_tagspace", pred_corpus_tagspace)
                    print("crf2train_corpus_tagspace", crf2train_corpus_tagspace)
                    print("crf2train_corpus_missing_tagspace", crf2train_corpus_missing_tagspace)
                    print("crf2pred_missing_tagspace", crf2pred_missing_tagspace)
                    print()
                    print()

                # global prediction
                # use each trainning brach to make a local predict and store in local_pred_tags
                # use you GLOBAL strategy to combine
                local_pred_tags = [] # store the prediction for each brach
                crf_training_corpus = []
                for crf_idx, curr_pred_missing_tagspace in crf2pred_missing_tagspace.items():

                    pred_unique_tagspace = pred_corpus_tagspace - crf2train_corpus_tagspace[crf_idx]
                    filtered_gold_labels = gold_labels
                    # filter the gold tags outside the training scope
                    if pred_unique_tagspace:
                        #print("Tagspace outside training scopre, ", pred_unique_tagspace)
                        filtered_gold_labels = [[token_label if token_label not in pred_unique_tagspace else 'O' for token_label in sent_labels] for sent_labels in gold_labels]
                        
                    for pt, fgl in zip(pred_tokens, filtered_gold_labels):
                        assert len(pt) == len(fgl), "inconsisnt sent len after filter\n {}\n {}".format(pt, fgl)

                    #print("pred_missing_tagspace", curr_pred_missing_tagspace)
                    if args.post:
                        curr_pred_missing_tagspace = set()
                    pred_missing_tagspace = sorted([tag2idx[untag] for untag in curr_pred_missing_tagspace]) 
                    pred_missing_tagspace = [pred_missing_tagspace] * len(filtered_gold_labels)

                    dataloader = build_dataloader(pred_tokens, filtered_gold_labels, 50, pred_missing_tagspace, 0, tag2idx, chr2idx, token2idx, train_args['caseless'], shuffle=False, drop_last=False)

                    # choose to eval the local prediction on the training brach
                    
                    # unzip batch and group by sents
                    merge_batch = True

                    #pred_tags = predictor.predict(ner_model, dataloader, crf_idx, merge_batch=merge_batch, totag=False)                
                    pred_tags = predictor.predict(ner_model, dataloader, crf_idx, args.pred_method, merge_batch=True, totag=True)
                    pred_tags = sorted(pred_tags, key=lambda item:item[0])
                    reorder_idx, pred_tags = zip(*pred_tags)
                    for i, re_idx in enumerate(sorted(reorder_idx)):
                        assert i == re_idx

                    if merge_batch:
                        assert len(pred_tokens) == len(pred_tags), "{} {}".format(len(pred_tokens), len(pred_tags))
                    else:
                        num_sample = sum(map(lambda t: len(t), dataloader)) 
                        assert num_sample == len(pred_tags), "{} {}".format(len(pred_tokens), len(pred_tags))
                    train_corpus = [train_args["train_file"][i].split("/")[-2] for i in crf2corpus[crf_idx]]
                    crf_training_corpus.append(train_corpus)

                    # store the prediction of current brach
                    local_pred_tags.append(pred_tags)
                
                # gloabl prediction with result from different brach
                global_pred, collison = global_predict(local_pred_tags)
                if args.show_collison:
                    display_collison(collison, pred_tokens, gold_labels)

                if args.verbose >= 2:
                    print("Number of different pred results", len(local_pred_tags))
                    print("Training brach and correspon training corpus", crf_training_corpus)
                
                show_pred_tagspace = True
                # the performance of current prediction on its training brach
                if args.global_eval:
                    print("Copus Level: ")
                    for crf_pred, crf_corpus in zip(local_pred_tags, crf_training_corpus):
                        print("Train on: ", crf_corpus)
                        evaluator.eval_w_pred_tag(crf_pred, gold_labels, show_pred_tagspace, 1)
                        checker = []
                        for crfp, gp in zip(crf_pred, global_pred):
                            checker.append(crfp == gp)
                        print("= gloabl pred? ", all(checker))
                        print()
                    print()

                revised_global_pred = []
                for sent in global_pred:
                    sent_pred = []
                    for token in sent:
                        if token in pred_corpus_tagspace:
                            sent_pred.append(token)
                        else:
                            sent_pred.append("O")
                    revised_global_pred.append(sent_pred)

                # global evaluation
                print(file_name)
                print("Global: ")
                if not args.post:
                    
                    if args.eval_filtered:
                        evaluator.eval_w_pred_tag(global_pred, filtered_gold_labels, args.verbose == 1, 1)
                    else:
                        evaluator.eval_w_pred_tag(global_pred, gold_labels, args.verbose == 1, 1)
                else:
                    if args.eval_filtered:
                        evaluator.eval_w_pred_tag(revised_global_pred, filtered_gold_labels, args.verbose == 1, 1)
                    else:
                        evaluator.eval_w_pred_tag(revised_global_pred, gold_labels, args.verbose == 1, 1)
                print()
                """
                error = []
                local_pred_tags_groupby_sent = list(zip(*local_pred_tags))
                for i, (p, g) in enumerate(zip(global_pred, gold_labels)):
                    assert len(p) == len(g), "inconsisnt sent len with global_pred and gold\n {}\n {}".format(p, g)
                    if p != g:
                        error.append((i, p, g, local_pred_tags_groupby_sent[i]))
                
                for sent_idx, prd, gld, local in error[:100]:
                    print("Sent ID: {}, len(Pred): {}, len(Gold): {}, len(locals): {} ".format(sent_idx, len(pred_tokens[sent_idx]), len(prd), len(gld)), list(map(len, local)))
                    print("Tokens: ", pred_tokens[sent_idx])
                    print("Gold: ", gld)
                    print("Global: ", prd)
                    print()
                    print("Gold: ", Counter(gld))
                    print("Global: ", Counter(prd))
                    print()
                """
                if args.annotate:
                    with open("./sample.tsv", "w") as outfile:
                        for i, sent in enumerate(zip(pred_tokens, global_pred)):
                            for t, tg in zip(*sent):
                                outfile.write("{}\t{}\n".format(t, tg))
                            outfile.write("\n")

                if args.debug_errors:
                    error = []
                    local_pred_tags_groupby_sent = list(zip(*local_pred_tags))
                    for i, (p, g) in enumerate(zip(global_pred, gold_labels)):
                        assert len(p) == len(g), "inconsisnt sent len with global_pred and gold\n {}\n {}".format(p, g)
                        if p != g:
                            pos = []
                            for k, (x, y) in enumerate(zip(p, g)):
                               
                                if x != y:
                                    pos.append(k)
                            error.append((i, p, g, local_pred_tags_groupby_sent[i], pos))
                    for sent_idx, prd, gld, local, pos in error[:args.error_num]:
                        print("Sent ID: {}, len(Pred): {}, len(Gold): {}, pos: {} ".format(sent_idx, len(prd), len(gld), pos))
                        

                        for i, token in enumerate(zip(pred_tokens[sent_idx], gld, prd)):
                            print("\t".join((str(i), ) + token))
                        print()

                print(global_pred[284])



                        #print("Tokens: ", pred_tokens[sent_idx])
                        #print("Gold: ", gld)
                        #print()
                        #print("Global: ", prd)
                        #for i, lcl in enumerate(local):
                        #    print("Local{}: ".format(i), lcl)
                        #print()
            else:
                raise NotImplementedError




        
