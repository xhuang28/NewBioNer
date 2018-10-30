import time
import torch.nn as nn
import model.utils as utils
from collections import Counter
import random
from tqdm import tqdm
import itertools
import sys
from copy import deepcopy
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer(object):
    def __init__(self, ner_model, packer, crit_ner, crit_lm, optimizer, evaluator, crf2corpus, plateau=False):
        self.ner_model = ner_model
        self.crit_ner = crit_ner
        self.packer = packer
        self.crit_lm = crit_lm
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.crf2corpus = crf2corpus
        self.sample_cnter = Counter()

        self.best_f1 = [float('-inf') for i in range(len(self.crf2corpus))]
        self.best_pre = [float('-inf') for i in range(len(self.crf2corpus))]
        self.best_rec = [float('-inf') for i in range(len(self.crf2corpus))]
        # [[F1,P,R], [F1, P, R]]
        self.corpus_best_vec = [[float('-inf')] * 3 for i in range(sum(map(len, self.crf2corpus.values())))]
        self.best_epoch_idx = float('-inf')
        self.best_state_dict = self.ner_model.state_dict()
        self.best_checkpoint_name = ""
        self.corpus_best_checkpoint_name = ["" for i in range(sum(map(len, self.crf2corpus.values())))]

        self.patience_count = 0
        self.training_time = 0
        self.track_list = list()

        self.plateau = plateau
        if plateau:
            self.scheduler = ReduceLROnPlateau(optimizer, 'min')

    def train(self, crf2train_dataloader, crf2dev_dataloader, dev_dataset_loader, epoch_list, args):
        start_time = time.time()
        
        for epoch_idx in epoch_list:
            args.start_epoch = epoch_idx
            curr_start_time = time.time()
            ###########################
            crf_no = random.randint(0, len(self.crf2corpus) - 1)
            ###########################
            
            cur_dataset = crf2train_dataloader[crf_no]
            epoch_loss = self.train_epoch(cur_dataset, crf_no, self.crit_ner, self.optimizer, args)

            # main evaluation on the combined dev in N21 or single dev in N2N
            corpus_name = [args.dev_file[i].split("/")[-2] for i in self.crf2corpus[crf_no]]
            print(args.dispatch, "Dev Corpus: ", corpus_name)

            dev_f1, dev_pre, dev_rec, dev_acc = self.eval_epoch(crf2dev_dataloader[crf_no], crf_no, args)

            if_add_patience = True
            if dev_f1 > self.best_f1[crf_no]:
                print("Prev Best F1: {:.4f} Curr Best F1: {:.4f}".format(self.best_f1[crf_no], dev_f1))
                self.best_epoch_idx = epoch_idx
                self.patience_count = 0
                self.best_f1[crf_no] = dev_f1
                self.best_pre[crf_no] = dev_pre
                self.best_rec[crf_no] = dev_rec
                self.best_state_dict = deepcopy(self.ner_model.state_dict())

                checkpoint_name = args.checkpoint + "/"
                checkpoint_name += args.dispatch + "_"  
                if args.dispatch in ["N2K", "N2N"]:
                    checkpoint_name += args.train_file[self.crf2corpus[crf_no][0]].split("/")[-2] + "_"
                checkpoint_name += "{:.4f}_{:.4f}_{:.4f}_{:d}".format(dev_f1, dev_pre, dev_rec, epoch_idx)

                print("NOW SAVING, ", checkpoint_name)
                print()


                self.drop_check_point(checkpoint_name, args)
                self.best_checkpoint_name = checkpoint_name

                if_add_patience &= False
            else:
                if args.dispatch == "N2N" or not args.stop_on_single:
                    self.patience_count += 1
            self.track_list.append({'loss': epoch_loss, 'dev_f1': dev_f1, 'dev_acc': dev_acc})

            if epoch_idx == args.epoch-1:
                last_checkpoint_name = args.checkpoint + "/"
                last_checkpoint_name += args.dispatch + "_"  
                if args.dispatch in ["N2K", "N2N"]:
                    last_checkpoint_name += args.train_file[self.crf2corpus[crf_no][0]].split("/")[-2] + "_"
                last_checkpoint_name += "LAST"+"_"
                last_checkpoint_name += "{:.4f}_{:.4f}_{:.4f}_{:d}".format(dev_f1, dev_pre, dev_rec, epoch_idx)

                print("NOW SAVING LAST, ", last_checkpoint_name)
                self.drop_check_point(last_checkpoint_name, args)
                print()
                
            # save check point for each corpus 
            if args.dispatch in ["N21", "N2K"]:
                print("Drop the best check point for single corpus")
                
                for cid in self.crf2corpus[crf_no]:
                    print(args.dev_file[cid])
                    cid_f1, cid_pre, cid_rec, cid_acc = self.eval_epoch(dev_dataset_loader[cid], crf_no, args)
                    # F1
                    if cid_f1 > self.corpus_best_vec[cid][0]:
                        print("Prev Best F1: {:.4f} Curr Best F1: {:.4f}".format(self.corpus_best_vec[cid][0], cid_f1))
                        self.corpus_best_vec[cid] = [cid_f1, cid_pre, cid_rec]
                        
                        if args.stop_on_single:
                            self.patience_count = 0
                        
                        checkpoint_name = args.checkpoint + "/"
                        checkpoint_name += args.dispatch + "_"  
                        checkpoint_name += args.dev_file[cid].split("/")[-2] + "_"
                        checkpoint_name += "{:.4f}_{:.4f}_{:.4f}_{:d}".format(cid_f1, cid_pre, cid_rec, epoch_idx)

                        print("NOW SAVING, ", checkpoint_name)
                        self.drop_check_point(checkpoint_name, args)
                        print()
                        self.corpus_best_checkpoint_name[cid] = checkpoint_name

                        if_add_patience &= False
                    else:
                        if_add_patience &= True
                if if_add_patience and args.stop_on_single:
                    self.patience_count += 1

            operating_time = time.time() - start_time
            h = operating_time // 3600
            m = (operating_time - 3600 * h) // 60
            s = operating_time - 3600 * h - 60 * m

            print("Epoch: [{:d}/{:d}]\t Patient: {:d}\t Current: {:.2f}\t Total: {:2d}:{:2d}:{:.2f}\n".format(args.start_epoch, args.epoch-1, self.patience_count, time.time() - curr_start_time, int(h), int(m), s))
            if self.patience_count >= args.patience and args.start_epoch >= args.least_iters:
                break

            # update lr
            if self.plateau:
                self.scheduler.step(dev_f1)
            else:
                utils.adjust_learning_rate(self.optimizer, args.lr / (1 + (args.start_epoch + 1) * args.lr_decay))

        print("Sample Frequence")
        for crf, corpus_idx in self.crf2corpus.items():
            corpus_name = [args.train_file[i].split("/")[-2] for i in corpus_idx]
            print(crf, corpus_name, self.sample_cnter[crf])
        print()


    def train_epoch(self, cur_dataset, crf_no, crit_ner, optimizer, args):
        #cur_dataset = crf2train_dataloader[crf_no]
        
        self.ner_model.train()
        epoch_loss = 0

        num_sample = sum(map(lambda t: len(t), cur_dataset)) 
        
        train_corpus = [args.train_file[i].split("/")[-2] for i in self.crf2corpus[crf_no]]
        print("Epoch: [{:d}/{:d}]".format(args.start_epoch, args.epoch - 1))
        print("Train corpus: ", train_corpus)
        
        if args.idea[:2] != 'P2':
            data_iter = itertools.chain.from_iterable(cur_dataset)
        else:
            data_iter = iter(cur_dataset)
        
        for f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v, corpus_mask_v, reorder in tqdm(
            data_iter, mininterval=2,
            desc=' - Total it %d' % (num_sample), leave=False, file=sys.stdout):
            
            if args.idea[:2] != 'P2':
                f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, corpus_mask_v = self.packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v, corpus_mask_v)
            else:
                if args.idea == 'P23':
                    proba_dist, tg_v = tg_v
                f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v, corpus_mask_v, reorder = f_f.cuda(), f_p.cuda(), b_f.cuda(), b_p.cuda(), w_f.cuda(), tg_v.cuda(), mask_v.cuda(), len_v.cuda(), corpus_mask_v.cuda(), reorder.cuda()
            
            self.ner_model.zero_grad()
            scores = self.ner_model(f_f, f_p, b_f, b_p, w_f, crf_no, corpus_mask_v)
            
            if args.idea == 'P23':
                loss = crit_ner(scores, [proba_dist, tg_v], mask_v, corpus_mask_v, idea = args.idea, sigmoid = args.sigmoid, mask_value = args.mask_value)
            else:
                loss = crit_ner(scores, tg_v, mask_v, corpus_mask_v, idea = args.idea, sigmoid = args.sigmoid, mask_value = args.mask_value)

            epoch_loss += utils.to_scalar(loss)
            if args.co_train:
                cf_p = f_p[0:-1, :].contiguous()
                cb_p = b_p[1:, :].contiguous()
                cf_y = w_f[1:, :].contiguous()
                cb_y = w_f[0:-1, :].contiguous()
                cfs, _ = self.ner_model.word_pre_train_forward(f_f, cf_p)
                loss = loss + args.lambda0 * self.crit_lm(cfs, cf_y.view(-1))
                cbs, _ = self.ner_model.word_pre_train_backward(b_f, cb_p)
                loss = loss + args.lambda0 * self.crit_lm(cbs, cb_y.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm(self.ner_model.parameters(), args.clip_grad)
            optimizer.step()

        epoch_loss = epoch_loss / num_sample
        self.sample_cnter[crf_no] += 1

        print("training loss: {:.4f}".format(epoch_loss))
        return epoch_loss


    def eval_epoch(self, dataloader, crf_no, args, show_pred_tagspace=False):
        if show_pred_tagspace:
            f1, pre, rec, acc, vloss, pred_cnter, gold_cnter = self.evaluator.eval_one_corpus(self.ner_model, dataloader, crf_no, None, show_pred_tagspace)
            print("P: {:.4f} R: {:.4f} F1: {:.4f} vloss: {:.6f}".format(pre, rec, f1, vloss))
            print("Pred: ", pred_cnter) 
            print("Gold: ", gold_cnter)
        else:
            f1, pre, rec, acc, vloss = self.evaluator.eval_one_corpus(self.ner_model, dataloader, crf_no, None, show_pred_tagspace)
            print("P: {:.4f} R: {:.4f} F1: {:.4f} vloss: {:.6f}".format(pre, rec, f1, vloss))
        return f1, pre, rec, acc

    def eval_batch_corpus(self, corpus_dataloaders, corpus_names, corpus2crf, use_best=True):
        curr_state_dict = deepcopy(self.ner_model.state_dict())
        if use_best:
            print("best epoch: ", self.best_epoch_idx)
            print("best checkpoint", self.best_checkpoint_name)            
            self.ner_model.load_state_dict(self.best_state_dict)
        self.evaluator.eval_batch_corpus(self.ner_model, corpus2crf, corpus_names, corpus_dataloaders, None, show_pred_tagspace=True)
        print("corpus_best_checkpoint_name", self.corpus_best_checkpoint_name) 
        if use_best:
            self.ner_model.load_state_dict(curr_state_dict)

    def drop_check_point(self, checkpoint_name, args):
        try:
            utils.save_checkpoint({
                'epoch': args.start_epoch,
                'state_dict': self.ner_model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, {'track_list': self.track_list,
                'args': vars(args)
                }, checkpoint_name)
            self.checkpoint_name = checkpoint_name
        except Exception as inst:
            print(inst)
