"""
.. module:: evaluator
    :synopsis: evaluation method (f1 score and accuracy)
 
.. moduleauthor:: Liyuan Liu, Frank Xu
"""


import torch
import numpy as np
import itertools

import model.utils as utils
from torch.autograd import Variable
from collections import Counter
from model.crf import CRFDecode_vb
from tqdm import tqdm
import sys

class eval_batch:
    """Base class for evaluation, provide method to calculate f1 score and accuracy 

    args: 
        packer: provide method to convert target into original space [TODO: need to improve] 
        l_map: dictionary for labels    
    """
   

    def __init__(self, packer, l_map):
        self.packer = packer
        self.l_map = l_map
        self.r_l_map = utils.revlut(l_map)

    def reset(self):
        """
        re-set all states
        """
        self.correct_labels = 0
        self.total_labels = 0
        self.gold_count = 0
        self.guess_count = 0
        self.overlap_count = 0

        self.pred_cnter = Counter()
        self.gold_cnter = Counter()

    def calc_f1_batch(self, decoded_data, target_data):
        """
        update statics for f1 score

        args: 
            decoded_data (batch_size, seq_len): prediction sequence
            target_data (batch_size, seq_len): ground-truth
        """
        batch_decoded = torch.unbind(decoded_data, 1)        
        batch_targets = torch.unbind(target_data, 0)

        for decoded, target in zip(batch_decoded, batch_targets):
            gold = self.packer.convert_for_eval(target)

            # remove padding
            length = utils.find_length_from_labels(gold, self.l_map)
            gold = gold[:length]
            best_path = decoded[:length]

            correct_labels_i, total_labels_i, gold_count_i, guess_count_i, overlap_count_i = self.eval_instance(best_path.cpu().numpy(), gold.cpu().numpy())
            self.correct_labels += correct_labels_i
            self.total_labels += total_labels_i
            self.gold_count += gold_count_i
            self.guess_count += guess_count_i
            self.overlap_count += overlap_count_i

    def calc_acc_batch(self, decoded_data, target_data):
        """
        update statics for accuracy

        args: 
            decoded_data (batch_size, seq_len): prediction sequence
            target_data (batch_size, seq_len): ground-truth
        """
        batch_decoded = torch.unbind(decoded_data, 1)
        batch_targets = torch.unbind(target_data, 0)

        for decoded, target in zip(batch_decoded, batch_targets):
            gold = self.packer.convert_for_eval(target)
            # remove padding
            length = utils.find_length_from_labels(gold, self.l_map)
            gold = gold[:length].numpy()
            best_path = decoded[:length].numpy()

            self.total_labels += length
            self.correct_labels += np.sum(np.equal(best_path, gold))

    def f1_score(self):
        """
        calculate f1 score based on statics
        """
        if self.guess_count == 0:
            return 0.0, 0.0, 0.0, 0.0
        precision = self.overlap_count / float(self.guess_count)
        recall = self.overlap_count / float(self.gold_count)
        if precision == 0.0 or recall == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        f = 2 * (precision * recall) / (precision + recall)
        accuracy = float(self.correct_labels) / self.total_labels
        return f, precision, recall, accuracy

    def acc_score(self):
        """
        calculate accuracy score based on statics
        """
        if 0 == self.total_labels:
            return 0.0
        accuracy = float(self.correct_labels) / self.total_labels
        return accuracy        

    def eval_instance(self, best_path, gold):
        """
        update statics for one instance

        args: 
            best_path (seq_len): predicted
            gold (seq_len): ground-truth
        """
        total_labels = len(best_path)
        correct_labels = np.sum(np.equal(best_path, gold))
        gold_chunks = utils.iobes_to_spans(gold, self.r_l_map)
        gold_count = len(gold_chunks)

        guess_chunks = utils.iobes_to_spans(best_path, self.r_l_map)
        guess_count = len(guess_chunks)



        # tests code
        self.gold_cnter += Counter([self.r_l_map[idx] for idx in gold])
        self.pred_cnter += Counter([self.r_l_map[idx] for idx in best_path])
        ############


        overlap_chunks = gold_chunks & guess_chunks
        overlap_count = len(overlap_chunks)

        return correct_labels, total_labels, gold_count, guess_count, overlap_count

    def eval_sent(self, pred, gold):
        """
        update statics for one instance

        args: 
            pred (seq_len): predicted
            gold (seq_len): ground-truth
        """
        total_labels = len(pred)
        correct_labels = sum(p==g for p, g in zip(pred, gold))
        gold_chunks = utils.iobes2spans(gold)
        gold_count = len(gold_chunks)

        guess_chunks = utils.iobes2spans(pred)
        guess_count = len(guess_chunks)

        # tests code
        self.gold_cnter += Counter(gold)
        self.pred_cnter += Counter(pred)
        ############

        overlap_chunks = gold_chunks & guess_chunks
        overlap_count = len(overlap_chunks)

        return correct_labels, total_labels, gold_count, guess_count, overlap_count

    def calc_f1_tag(self, pred, gold):
        correct_labels_i, total_labels_i, gold_count_i, guess_count_i, overlap_count_i = self.eval_sent(pred, gold)
        self.correct_labels += correct_labels_i
        self.total_labels += total_labels_i
        self.gold_count += gold_count_i
        self.guess_count += guess_count_i
        self.overlap_count += overlap_count_i

class eval_wc(eval_batch):
    """evaluation class for LM-LSTM-CRF

    args: 
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
        score_type: use f1score with using 'f'

    """
   
    def __init__(self, packer, l_map, score_type):
        eval_batch.__init__(self, packer, l_map)

        self.decoder = CRFDecode_vb(len(l_map), l_map['<start>'], l_map['<pad>'])

        if 'f' in score_type:
            self.eval_b = self.calc_f1_batch
            self.calc_s = self.f1_score
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def calc_score(self, ner_model, dataset_loader, crf_no, crit_ner, verbose=0):
        """
        calculate score for pre-selected metrics

        args: 
            ner_model: LM-LSTM-CRF model
            dataset_loader: loader class for test set
        """
        ner_model.eval()
        self.reset()

        epoch_loss = 0 
        for f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v, corpus_mask_v, reorder in itertools.chain.from_iterable(dataset_loader):
            f_f, f_p, b_f, b_p, w_f, _, mask_v, corpus_mask_v = self.packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v, corpus_mask_v, volatile=True)
            scores = ner_model(f_f, f_p, b_f, b_p, w_f, crf_no, corpus_mask_v)
            
            loss = crit_ner(scores, _, mask_v, corpus_mask_v)
            #print(loss)
            epoch_loss += utils.to_scalar(loss)
            
            decoded = self.decoder.decode(scores.data, mask_v.data)
            self.eval_b(decoded, tg)

        ###################
        # tests code
        print("validation loss: {}, {}".format(epoch_loss, epoch_loss / sum(map(lambda t: len(t), dataset_loader))))
        if verbose > 0:
            print("pred", self.pred_cnter)
            print("gold", self.gold_cnter)
        ############
        
        return self.calc_s()

class Evaluator(eval_batch):
    """evaluation class for LM-LSTM-CRF

    args: 
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
        score_type: use f1score with using 'f'

    """
   
    def __init__(self, predictor, packer, l_map, score_type, pred_method):
        eval_batch.__init__(self, packer, l_map)
        
        self.predictor = predictor
        self.pred_method = pred_method
        
        if 'f' in score_type:
            self.eval_b = self.calc_f1_batch
            self.calc_s = self.f1_score
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def eval_w_pred_tag(self, predict, gold, show_pred_tagspace, verbose=0):
        self.reset()
        for p, g in zip(predict, gold):
            self.calc_f1_tag(p, g)

        f1, pre, rec, acc = self.calc_s()
        if verbose > 0:
            if show_pred_tagspace:
                print("P: {:.4f} R: {:.4f} F1: {:.4f}".format(pre, rec, f1))
                print("Pred: ", self.pred_cnter) 
                print("Gold: ", self.gold_cnter) 
            else:
                print("P: {:.4f} R: {:.4f} F1: {:.4f}".format(pre, rec, f1))

        if show_pred_tagspace:
            return f1, pre, rec, acc, self.pred_cnter, self.gold_cnter
        else:
            return f1, pre, rec, acc


    def eval_one_corpus(self, ner_model, dataset_loader, crf_no, crit_ner, show_tagspace=False):
        """
        calculate score for pre-selected metrics

        args: 
            ner_model: LM-LSTM-CRF model
            dataset_loader: loader class for test set
        """
        ner_model.eval()
        self.reset()
        if crit_ner is not None:
            validation_loss = 0 
        else:
            validation_loss = float("-inf")
        num_sample = sum(map(lambda t: len(t), dataset_loader)) 
        for f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v, corpus_mask_v, reorder in tqdm(
            itertools.chain.from_iterable(dataset_loader), mininterval=2,
            desc=' - Total it %d' % (num_sample), leave=False, file=sys.stdout):
            f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, corpus_mask_v = self.packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v, corpus_mask_v, volatile=True)
            
            decoded, scores = self.predictor.predict_batch(ner_model, crf_no, f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v, corpus_mask_v, self.pred_method)
            if crit_ner is not None:
                loss = crit_ner(scores, tg_v, mask_v, corpus_mask_v)
                validation_loss += utils.to_scalar(loss)
            self.eval_b(decoded, tg)
        if crit_ner is not None:
            validation_loss = validation_loss / sum(map(lambda t: len(t), dataset_loader))
        if show_tagspace:
            return (*self.calc_s(), validation_loss, self.pred_cnter, self.gold_cnter)
        else:
            return (*self.calc_s(), validation_loss)

    def eval_corpus_same_crf(self, ner_model, crf_no, crf2corpus, corpus_names, dataloader, crit_ner, show_pred_tagspace=False):
        print("Eval on SAME CRF Brach")
        crf_res = []
        for cid in crf2corpus[crf_no]:
            print(corpus_names[cid])
            corpus_res = []
            if show_pred_tagspace:
                cid_f1, cid_pre, cid_rec, cid_acc, cid_vloss, cid_pred_cnter, cid_gold_cnter = self.eval_one_corpus(ner_model, dataloader[cid], crf_no, crit_ner, show_pred_tagspace)
                print("P: {:.4f} R: {:.4f} F1: {:.4f} vloss: {:.6f}".format(cid_pre, cid_rec, cid_f1, cid_vloss))
                print("Pred: ", cid_pred_cnter) 
                print("Gold: ", cid_gold_cnter) 
            else:
                cid_f1, cid_pre, cid_rec, cid_acc, cid_vloss = self.eval_one_corpus(ner_model, dataloader[cid], crf_no, crit_ner, show_pred_tagspace)
                print("P: {:.4f} R: {:.4f} F1: {:.4f} vloss: {:.6f}".format(cid_pre, cid_rec, cid_f1, cid_vloss))   
            corpus_res = [cid_f1, cid_pre, cid_rec, cid_acc]
            crf_res.append(corpus_res)
        print()
        return crf_res

    def eval_batch_corpus(self, ner_model, corpus2crf, corpus_names, corpus_dataloaders, crit_ner, show_pred_tagspace=False):
        print("Eval on Corpurar")
        for i in range(len(corpus_names)):
            dataloader = corpus_dataloaders[i]
            print(corpus_names[i])
            if show_pred_tagspace:
                f1, pre, rec, acc, vloss, pred_cnter, gold_cnter = self.eval_one_corpus(ner_model, dataloader, corpus2crf[i], crit_ner, show_pred_tagspace)
                print("P: {:.4f} R: {:.4f} F1: {:.4f} vloss: {:.6f}".format(pre, rec, f1, vloss))
                print("Pred: ", pred_cnter) 
                print("Gold: ", gold_cnter)
            else:
                f1, pre, rec, acc, vloss = self.eval_one_corpus(ner_model, dataloader, corpus2crf[i], crit_ner, show_pred_tagspace)
                print("P: {:.4f} R: {:.4f} F1: {:.4f} vloss: {:.6f}".format(pre, rec, f1, vloss))
            print()
        print()



