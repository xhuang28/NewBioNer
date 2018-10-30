"""
.. module:: crf
    :synopsis: conditional random field
 
.. moduleauthor:: Liyuan Liu
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.sparse as sparse
import model.utils as utils
from copy import deepcopy
import numpy as np

class CRF_L(nn.Module):
    """Conditional Random Field (CRF) layer. This version is used in Ma et al. 2016, has more parameters than CRF_S
 
    args: 
        hidden_dim : input dim size 
        tagset_size: target_set_size 
        if_biase: whether allow bias in linear trans    
    """
    

    def __init__(self, hidden_dim, tagset_size, if_bias=True, sigmoid=""):
        assert sigmoid
        super(CRF_L, self).__init__()
        self.sigmoid = sigmoid
        self.tagset_size = tagset_size
        self.transitions = nn.Linear(hidden_dim, self.tagset_size * self.tagset_size, bias=if_bias)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=if_bias)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

    def rand_init(self):
        """random initialization
        """
        utils.init_linear(self.hidden2tag)
        utils.init_linear(self.transitions)

    def forward(self, feats):
        """
        args: 
            feats (batch_size, seq_len, hidden_dim) : input score from previous layers
        return:
            output from crf layer (batch_size, seq_len, tag_size, tag_size)
        """
        ins_num = feats.size(0) * feats.size(1)
        scores = self.hidden2tag(feats).view(ins_num, self.tagset_size, 1).expand(ins_num, self.tagset_size, self.tagset_size)
        trans_ = self.transitions(feats).view(ins_num, self.tagset_size, self.tagset_size)
        
        if self.sigmoid == "nosig":
            return scores + trans_
        elif self.sigmoid == "relu":
            return self.ReLU(scores + trans_)


class CRF_S(nn.Module):
    """Conditional Random Field (CRF) layer. This version is used in Lample et al. 2016, has less parameters than CRF_L.

    args: 
        hidden_dim: input dim size
        tagset_size: target_set_size
        if_biase: whether allow bias in linear trans
 
    """
    
    def __init__(self, hidden_dim, tagset_size, if_bias=True):
        super(CRF_S, self).__init__()
        self.tagset_size = tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=if_bias)
        self.transitions = nn.Parameter(torch.Tensor(self.tagset_size, self.tagset_size))

    def rand_init(self):
        """random initialization
        """
        utils.init_linear(self.hidden2tag)
        self.transitions.data.zero_()

    def forward(self, feats):
        """
        args: 
            feats (batch_size, seq_len, hidden_dim) : input score from previous layers
        return:
            output from crf layer (batch_size, seq_len, tag_size, tag_size)
        """
        
        ins_num = feats.size(0)
        scores = self.hidden2tag(feats)
        crf_scores = scores.view(-1, self.tagset_size, 1).expand(ins_num, self.tagset_size, self.tagset_size) + self.transitions.view(1, self.tagset_size, self.tagset_size).expand(ins_num, self.tagset_size, self.tagset_size)

        return crf_scores

class CRFRepack:
    """Packer for word level model
    
    args:
        tagset_size: target_set_size
        if_cuda: whether use GPU
    """

    def __init__(self, tagset_size, if_cuda):
        
        self.tagset_size = tagset_size
        self.if_cuda = if_cuda

    def repack_vb(self, feature, target, mask):
        """packer for viterbi loss

        args: 
            feature (Seq_len, Batch_size): input feature
            target (Seq_len, Batch_size): output target
            mask (Seq_len, Batch_size): padding mask
        return:
            feature (Seq_len, Batch_size), target (Seq_len, Batch_size), mask (Seq_len, Batch_size)
        """
        
        if self.if_cuda:
            fea_v = autograd.Variable(feature.transpose(0, 1)).cuda()
            tg_v = autograd.Variable(target.transpose(0, 1)).unsqueeze(2).cuda()
            mask_v = autograd.Variable(mask.transpose(0, 1)).cuda()
        else:
            fea_v = autograd.Variable(feature.transpose(0, 1))
            tg_v = autograd.Variable(target.transpose(0, 1)).contiguous().unsqueeze(2)
            mask_v = autograd.Variable(mask.transpose(0, 1)).contiguous()
        return fea_v, tg_v, mask_v

    def repack_gd(self, feature, target, current):
        """packer for greedy loss

        args: 
            feature (Seq_len, Batch_size): input feature
            target (Seq_len, Batch_size): output target
            current (Seq_len, Batch_size): current state
        return:
            feature (Seq_len, Batch_size), target (Seq_len * Batch_size), current (Seq_len * Batch_size, 1, 1)
        """
        if self.if_cuda:
            fea_v = autograd.Variable(feature.transpose(0, 1)).cuda()
            ts_v = autograd.Variable(target.transpose(0, 1)).cuda().view(-1)
            cs_v = autograd.Variable(current.transpose(0, 1)).cuda().view(-1, 1, 1)
        else:
            fea_v = autograd.Variable(feature.transpose(0, 1))
            ts_v = autograd.Variable(target.transpose(0, 1)).contiguous().view(-1)
            cs_v = autograd.Variable(current.transpose(0, 1)).contiguous().view(-1, 1, 1)
        return fea_v, ts_v, cs_v

    def convert_for_eval(self, target):
        """convert target to original decoding

        args: 
            target: input labels used in training
        return:
            output labels used in test
        """
        return target % self.tagset_size


class CRFRepack_WC:
    """Packer for model with char-level and word-level

    args:
        tagset_size: target_set_size
        if_cuda: whether use GPU
        
    """

    def __init__(self, tagset_size, if_cuda):
        
        self.tagset_size = tagset_size
        self.if_cuda = if_cuda

    def repack_vb(self, f_f, f_p, b_f, b_p, w_f, target, mask, len_b, corpus_mask, volatile=False):
        """packer for viterbi loss

        args: 
            f_f (Char_Seq_len, Batch_size) : forward_char input feature 
            f_p (Word_Seq_len, Batch_size) : forward_char input position
            b_f (Char_Seq_len, Batch_size) : backward_char input feature
            b_p (Word_Seq_len, Batch_size) : backward_char input position
            w_f (Word_Seq_len, Batch_size) : input word feature
            target (Seq_len, Batch_size) : output target
            mask (Word_Seq_len, Batch_size) : padding mask
            len_b (Batch_size, 2) : length of instances in one batch
        return:
            f_f (Char_Reduced_Seq_len, Batch_size), f_p (Word_Reduced_Seq_len, Batch_size), b_f (Char_Reduced_Seq_len, Batch_size), b_p (Word_Reduced_Seq_len, Batch_size), w_f (size Word_Seq_Len, Batch_size), target (Reduced_Seq_len, Batch_size), mask  (Word_Reduced_Seq_len, Batch_size)

        """
        mlen, _ = len_b.max(0)
        mlen = mlen.squeeze()
        ocl = b_f.size(1)
        if self.if_cuda:
            f_f = autograd.Variable(f_f[:, 0:mlen[0]].transpose(0, 1), volatile=volatile).cuda()
            f_p = autograd.Variable(f_p[:, 0:mlen[1]].transpose(0, 1), volatile=volatile).cuda()
            b_f = autograd.Variable(b_f[:, -mlen[0]:].transpose(0, 1), volatile=volatile).cuda()
            b_p = autograd.Variable((b_p[:, 0:mlen[1]] - ocl + mlen[0]).transpose(0, 1), volatile=volatile).cuda()
            w_f = autograd.Variable(w_f[:, 0:mlen[1]].transpose(0, 1), volatile=volatile).cuda()
            tg_v = autograd.Variable(target[:, 0:mlen[1]].transpose(0, 1), volatile=volatile).unsqueeze(2).cuda()
            mask_v = autograd.Variable(mask[:, 0:mlen[1]].transpose(0, 1), volatile=volatile).cuda()
            corpus_mask_v = autograd.Variable(corpus_mask.repeat(mlen[1], 1, 1, 1), volatile=volatile).cuda()

        else:
            f_f = autograd.Variable(f_f[:, 0:mlen[0]].transpose(0, 1))
            f_p = autograd.Variable(f_p[:, 0:mlen[1]].transpose(0, 1))
            b_f = autograd.Variable(b_f[:, -mlen[0]:].transpose(0, 1))
            b_p = autograd.Variable((b_p[:, 0:mlen[1]] - ocl + mlen[0]).transpose(0, 1))
            w_f = autograd.Variable(w_f[:, 0:mlen[1]].transpose(0, 1))
            tg_v = autograd.Variable(target[:, 0:mlen[1]].transpose(0, 1)).unsqueeze(2)
            mask_v = autograd.Variable(mask[:, 0:mlen[1]].transpose(0, 1))
            corpus_mask_v = autograd.Variable(corpus_mask.repeat(mlen[1], 1, 1, 1))

        return f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, corpus_mask_v

    def convert_for_eval(self, target):
        """convert for eval

        args: 
            target: input labels used in training
        return:
            output labels used in test
        """
        return target % self.tagset_size


class CRFLoss_gd(nn.Module):
    """loss for greedy decode loss, i.e., although its for CRF Layer, we calculate the loss as 

    .. math::
        \sum_{j=1}^n \log (p(\hat{y}_{j+1}|z_{j+1}, \hat{y}_{j}))

    instead of 
    
    .. math::
        \sum_{j=1}^n \log (\phi(\hat{y}_{j-1}, \hat{y}_j, \mathbf{z}_j)) - \log (\sum_{\mathbf{y}' \in \mathbf{Y}(\mathbf{Z})} \prod_{j=1}^n \phi(y'_{j-1}, y'_j, \mathbf{z}_j) )

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch
    
    """

    def __init__(self, tagset_size, start_tag, end_tag, average_batch=True):
        super(CRFLoss_gd, self).__init__()
        self.tagset_size = tagset_size
        self.average_batch = average_batch
        self.crit = nn.CrossEntropyLoss(size_average=self.average_batch)

    def forward(self, scores, target, current):
        """
        args: 
            scores (Word_Seq_len, Batch_size, target_size_from, target_size_to): crf scores
            target (Word_Seq_len, Batch_size): golden list
            current (Word_Seq_len, Batch_size): current state
        return:
            crf greedy loss
        """
        ins_num = current.size(0)
        current = current.expand(ins_num, 1, self.tagset_size)
        scores = scores.view(ins_num, self.tagset_size, self.tagset_size)
        current_score = torch.gather(scores, 1, current).squeeze()
        return self.crit(current_score, target)

class CRFLoss_vb(nn.Module):
    """loss for viterbi decode

    .. math::
        \sum_{j=1}^n \log (\phi(\hat{y}_{j-1}, \hat{y}_j, \mathbf{z}_j)) - \log (\sum_{\mathbf{y}' \in \mathbf{Y}(\mathbf{Z})} \prod_{j=1}^n \phi(y'_{j-1}, y'_j, \mathbf{z}_j) )

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch
        
    """

    def __init__(self, tagset_size, start_tag, end_tag, average_batch=True, O_idx=0):
        super(CRFLoss_vb, self).__init__()
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.average_batch = average_batch
        self.O_idx = O_idx
    
    def calc_energy_gold_ts(self, scores, target, mask, corpus_mask):
        # calculate energy (unnormalized log proba) of the gold tag sequence
        seq_len = scores.size(0)
        bat_size = scores.size(1)
        tg_energy = torch.gather(scores.view(seq_len, bat_size, -1), 2, target).view(seq_len, bat_size)  # seq_len * bat_size
        tg_energy = tg_energy.masked_select(mask).sum()
        
        return tg_energy
    
    def forward_algo(self, scores, target, mask, corpus_mask):
        # Forward Algorithm
        seq_len = scores.size(0)
        bat_size = scores.size(1)
        
        seq_iter = enumerate(scores)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        cur_partition = inivalues[:, self.start_tag, :]  # bat_size * to_target_size
        partition = cur_partition
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # cur_partition: previous->current results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target            
            cur_values = cur_values + cur_partition.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)
            cur_partition = utils.log_sum_exp(cur_values, self.tagset_size)
                  # (bat_size * from_target * to_target) -> (bat_size * to_target)
            partition = utils.switch(partition, cur_partition,
                                     mask[idx].contiguous().view(bat_size, 1).expand(bat_size, self.tagset_size)).contiguous().view(bat_size, -1)
        
        #only need end at end_tag
        partition = partition[:, self.end_tag].sum()
        
        return partition
    
    def restricted_forward_algo_v1(self, scores, target, mask, corpus_mask, sigmoid):
        # Restricted Forward Algorithm v1
        # "O": Set scores of all local labels (not including "O") to 0
        # "NE": Set scores of all other labels to 0
        seq_len = scores.size(0)
        bat_size = scores.size(1)
        gold_labels = (target / 35).view(target.shape[0], target.shape[1])
        
        seq_iter = enumerate(scores)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        cur_partition = inivalues[:, self.start_tag, :]  # bat_size * to_target_size
        partition = cur_partition
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # cur_partition: previous->current results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            curr_labels = gold_labels[idx,:]
            
            # mask cur_partition and cur_values to rule out undesired tag sequences
            partition_mask = np.ones(cur_partition.shape)
            values_mask = np.ones(cur_values.shape)
            for i in range(partition_mask.shape[0]):
                curr_label = curr_labels[i].cpu().data.numpy()[0]
                if curr_label == self.O_idx:
                    idx_annotated = np.where(corpus_mask[0,i,0].data)[0]
                    idx_annotated = np.array([r for r in idx_annotated if r!=self.O_idx]) # exclude "O"
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
                cur_partition = utils.switch(neg_inf_partition, cur_partition, partition_mask).view(cur_partition.shape)
                cur_values = utils.switch(neg_inf_values, cur_values, values_mask).view(cur_values.shape)
            
            cur_values = cur_values + cur_partition.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)
            cur_partition = utils.log_sum_exp(cur_values, self.tagset_size)
                  # (bat_size * from_target * to_target) -> (bat_size * to_target)
            partition = utils.switch(partition.contiguous(), cur_partition.contiguous(),
                                     mask[idx].contiguous().view(bat_size, 1).expand(bat_size, self.tagset_size)).contiguous().view(bat_size, -1)
            
        #only need end at end_tag
        partition = partition[:, self.end_tag].sum()
        
        return partition
    
    def restricted_forward_algo_v2(self, scores, target, mask, corpus_mask, sigmoid, mask_value):
        # Restricted Forward Algorithm v1
        # "O": Set scores of all non-local labels to 0
        # "NE": No changes
        seq_len = scores.size(0)
        bat_size = scores.size(1)
        gold_labels = (target / 35).view(target.shape[0], target.shape[1])
        
        seq_iter = enumerate(scores)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        cur_partition = inivalues[:, self.start_tag, :]  # bat_size * to_target_size
        partition = cur_partition
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # cur_partition: previous->current results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            curr_labels = gold_labels[idx,:]
            
            # mask cur_partition and cur_values to rule out undesired tag sequences
            partition_mask, partition_mask_v = np.ones(cur_partition.shape), np.ones(cur_partition.shape)
            values_mask, values_mask_v = np.ones(cur_values.shape), np.ones(cur_values.shape)
            for i in range(partition_mask.shape[0]):
                curr_label = curr_labels[i].cpu().data.numpy()[0]
                if curr_label == self.O_idx:
                    idx_unannotated = np.where((1-corpus_mask[0,i,0]).data)[0]
                    partition_mask[i,idx_unannotated] = 0
                    partition_mask_v[i,idx_unannotated] = mask_value
                    values_mask[i,idx_unannotated,:] = 0
                    values_mask_v[i,idx_unannotated,:] = mask_value
            
            partition_mask = autograd.Variable(torch.FloatTensor(partition_mask)).cuda()
            partition_mask_v = autograd.Variable(torch.FloatTensor(partition_mask_v)).cuda()
            values_mask = autograd.Variable(torch.FloatTensor(values_mask)).cuda()
            values_mask_v = autograd.Variable(torch.FloatTensor(values_mask_v)).cuda()
            if sigmoid == "relu":
                cur_partition = cur_partition * partition_mask_v
                cur_values = cur_values * values_mask_v
            elif mask_value == 0:
                neg_inf_partition = autograd.Variable(torch.FloatTensor(np.full(cur_partition.shape, -1e9))).cuda()
                neg_inf_values = autograd.Variable(torch.FloatTensor(np.full(cur_values.shape, -1e9))).cuda()
                cur_partition = utils.switch(neg_inf_partition, cur_partition.contiguous(), partition_mask).view(cur_partition.shape)
                cur_values = utils.switch(neg_inf_values, cur_values.contiguous(), values_mask).view(cur_values.shape)
            else:
                neg_inf_partition = autograd.Variable(torch.FloatTensor(np.full(cur_partition.shape, np.log(mask_value)))).cuda()
                neg_inf_values = autograd.Variable(torch.FloatTensor(np.full(cur_values.shape, np.log(mask_value)))).cuda()
                cur_partition = utils.switch(neg_inf_partition, cur_partition.contiguous(), partition_mask).view(cur_partition.shape)
                cur_values = utils.switch(neg_inf_values, cur_values.contiguous(), values_mask).view(cur_values.shape)
            
            cur_values = cur_values + cur_partition.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)
            cur_partition = utils.log_sum_exp(cur_values, self.tagset_size)
                  # (bat_size * from_target * to_target) -> (bat_size * to_target)
            partition = utils.switch(partition, cur_partition,
                                     mask[idx].contiguous().view(bat_size, 1).expand(bat_size, self.tagset_size)).contiguous().view(bat_size, -1)
            
        #only need end at end_tag
        partition = partition[:, self.end_tag].sum()
        
        return partition
    
    def restricted_forward_algo_v3(self, scores, target, mask, corpus_mask, sigmoid, proba_dist):
        # Restricted Forward Algorithm v1
        # "O": Set scores of all local labels (not including "O") to 0
        # "NE": Set scores of all other labels to 0
        seq_len = scores.size(0)
        bat_size = scores.size(1)
        gold_labels = (target / 35).view(target.shape[0], target.shape[1])
        
        seq_iter = enumerate(scores)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        cur_partition = inivalues[:, self.start_tag, :]  # bat_size * to_target_size
        partition = cur_partition
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # cur_partition: previous->current results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            curr_labels = gold_labels[idx,:]
            
            # mask cur_partition and cur_values to rule out undesired tag sequences
            partition_mask = np.ones(cur_partition.shape)
            values_mask = np.ones(cur_values.shape)
            for i in range(partition_mask.shape[0]):
                curr_label = curr_labels[i].cpu().data.numpy()[0]
                if curr_label == self.O_idx:
                    idx_annotated = np.where(corpus_mask[0,i,0].data)[0]
                    idx_annotated = np.array([r for r in idx_annotated if r!=self.O_idx]) # exclude "O"
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
                cur_values = utils.switch(neg_inf_values, cur_values, values_mask).view(cur_values.shape)
            
            curr_proba_dist = proba_dist[idx-1]
            cur_partition = cur_partition + autograd.Variable(torch.FloatTensor(curr_proba_dist)).cuda()
            cur_values = cur_values + cur_partition.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)
            cur_partition = utils.log_sum_exp(cur_values, self.tagset_size)
                  # (bat_size * from_target * to_target) -> (bat_size * to_target)
            partition = utils.switch(partition.contiguous(), cur_partition.contiguous(),
                                     mask[idx].contiguous().view(bat_size, 1).expand(bat_size, self.tagset_size)).contiguous().view(bat_size, -1)
            
        #only need end at end_tag
        partition = partition[:, self.end_tag].sum()
        
        return partition
    
    
    def forward(self, scores, target, mask, corpus_mask, idea=None, sigmoid="", mask_value=None):
        """
        args:
            scores (seq_len, bat_size, target_size_from, target_size_to) : crf scores
            target (seq_len, bat_size, 1) : golden state
            mask (size seq_len, bat_size) : mask for padding
            *idea ("Li", "P11", "P12", "P21"...): idea for training (loss calculation)
        return:
            loss
        """
        assert sigmoid
        assert mask_value is not None
        bat_size = scores.size(1)
        
        # numerator and denominator: ...of the likelihood function:)
        
        # Global training (Phase 2)
        if idea == "P10":
            numerator = self.calc_energy_gold_ts(scores, target, mask, corpus_mask)
            denominator = self.forward_algo(scores, target, mask, corpus_mask)
        # Li's masking approach
        elif idea == "Li":
            if sigmoid == "relu":
                scores = scores * corpus_mask
            else:
                neg_inf_scores = autograd.Variable(torch.FloatTensor(np.full(scores.shape, -1e9))).cuda()
                scores = utils.switch(neg_inf_scores, scores, corpus_mask).view(scores.shape)
            numerator = self.calc_energy_gold_ts(scores, target, mask, corpus_mask)
            denominator = self.forward_algo(scores, target, mask, corpus_mask)
        elif idea == "P11":
            numerator = self.restricted_forward_algo_v1(scores, target, mask, corpus_mask, sigmoid)
            denominator = self.forward_algo(scores, target, mask, corpus_mask)
        elif idea == "P12":
            numerator = self.calc_energy_gold_ts(scores, target, mask, corpus_mask)
            denominator = self.restricted_forward_algo_v2(scores, target, mask, corpus_mask, sigmoid, mask_value)
        elif idea == "P22":
            numerator = self.calc_energy_gold_ts(scores, target, mask, corpus_mask)
            denominator = self.forward_algo(scores, target, mask, corpus_mask)
        elif idea == "P23":
            proba_dist, target = target
            numerator = self.restricted_forward_algo_v3(scores, target, mask, corpus_mask, sigmoid, proba_dist)
            denominator = self.forward_algo(scores, target, mask, corpus_mask)
        else:
            print("\n\n**********Idea not implemented!**********\n\n")
            assert False
        
        # average_batch
        if self.average_batch:
            loss = (denominator - numerator) / bat_size
        else:
            loss = (denominator - numerator)
        return loss

'''
class CRFLoss_sf(nn.Module):
    """loss for viterbi decode

    .. math::
        \sum_{j=1}^n \log (\phi(\hat{y}_{j-1}, \hat{y}_j, \mathbf{z}_j)) - \log (\sum_{\mathbf{y}' \in \mathbf{Y}(\mathbf{Z})} \prod_{j=1}^n \phi(y'_{j-1}, y'_j, \mathbf{z}_j) )

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch
        
    """

    def __init__(self, tagset_size, start_tag, end_tag, cost_value=1, change_gold=False, change_prob=0.9, average_batch=True):
        super(CRFLoss_sf, self).__init__()
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.average_batch = average_batch
        self.cost_value = cost_value
        print("SF, Cost Value {:.2f}".format(self.cost_value))

    def forward(self, scores, target, mask, corpus_mask):
        """
        args:
            scores (seq_len, bat_size, target_size_from, target_size_to) : crf scores
            target (seq_len, bat_size, 1) : golden state
            mask (size seq_len, bat_size) : mask for padding
        return:
            loss
        """
        # print(scores)
        # print(target)
        # print(mask)

        # calculate batch size and seq len
        seq_len = scores.size(0)
        bat_size = scores.size(1)

        scores = scores * corpus_mask
        argumented_cost = self.build_argumented_cost(scores.data, target, self.cost_value)
        scores = scores + autograd.Variable(argumented_cost)
        # calculate sentence score
        tg_energy = torch.gather(scores.view(seq_len, bat_size, -1), 2, target).view(seq_len, bat_size)  # seq_len * bat_size
        tg_energy = tg_energy.masked_select(mask).sum()

        # calculate forward partition score

        # build iter
        seq_iter = enumerate(scores)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.start_tag, :]  # bat_size * to_target_size
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)
            cur_partition = utils.log_sum_exp(cur_values, self.tagset_size)
                  # (bat_size * from_target * to_target) -> (bat_size * to_target)
            partition = utils.switch(partition, cur_partition,
                                     mask[idx].contiguous().view(bat_size, 1).expand(bat_size, self.tagset_size)).contiguous().view(bat_size, -1)
            # the following two may achieve higher speed, but raise run-time error
            # new_partition = partition.clone()
            # new_partition.masked_scatter_(mask[idx].view(-1, 1).expand(bat_size, self.tagset_size), cur_partition)  #0 for partition, 1 for cur_partition
            # partition = new_partition
            
        #only need end at end_tag
        partition = partition[:, self.end_tag].sum()
        # average = mask.sum()

        # average_batch
        if self.average_batch:
            loss = (partition - tg_energy) / bat_size
        else:
            loss = (partition - tg_energy)
        return loss

    def build_cost_mask(self, trellis, target, cost_value):
        unshift_target = target % self.tagset_size
        cost_mask = trellis.new(trellis.size()).fill_(cost_value)

        # need a fancy way
        for row in range(len(cost_mask)):
            curr_tag = utils.to_scalar(unshift_target[row])
            cost_mask[row][:,curr_tag] = 0
        return cost_mask

    def build_argumented_cost(self, trellis, target, cost_value):
        unshift_target = target % self.tagset_size
        argumented_cost = [self.build_cost_mask(trellis[0], target[0], 0.0)]
        for i in range(1, len(trellis)):
            cost = self.build_cost_mask(trellis[i], target[i], self.cost_value)
            argumented_cost.append(cost)
        return torch.stack(argumented_cost)


class CRFLoss_sf(nn.Module):
    """loss for max_margin

    .. math::
        \sum_{j=1}^n \log (\phi(\hat{y}_{j-1}, \hat{y}_j, \mathbf{z}_j)) - \log (\sum_{\mathbf{y}' \in \mathbf{Y}(\mathbf{Z})} \prod_{j=1}^n \phi(y'_{j-1}, y'_j, \mathbf{z}_j) )

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch
        
    """

    def __init__(self, tagset_size, start_tag, end_tag, o_tag, cost_value=1.0, change_gold=False, change_prob=0.9, average_batch=True):
        super(CRFLoss_sf, self).__init__()
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.o_tag = o_tag
        self.end_tag = end_tag
        self.average_batch = average_batch
        self.cost_value = cost_value
        self.change_gold = change_gold
        self.change_prob = change_prob

        print("SF, Cost Value {:.2f}, Change Gold: {}, Prob: {:.2f}".format(self.cost_value, self.change_gold, self.change_prob))

    def forward(self, trellis, gold, mask, corpus_mask):
        """
        args:
            trellis (seq_len, bat_size, target_size_from, target_size_to) : crf trellis
            gold (seq_len, bat_size, 1) : golden state
            mask (size seq_len, bat_size) : mask for padding
        return:
            loss
        """
        # calculate batch size and seq len
        seq_len = trellis.size(0)
        bat_size = trellis.size(1)
        
        # calculate sentence score
        
        argumented_cost = self.build_argumented_cost(trellis.data, gold, self.cost_value)
        trellis = trellis + autograd.Variable(argumented_cost)
        
        # predict on modified trellis

        preds = self.decode(trellis.data, mask.data)  
        shifted_preds = self.shift_pred(preds, seq_len, bat_size)

        pred_energy = self.calculate_energy(shifted_preds, trellis, mask)

        if self.change_gold:
            prob = torch.Tensor(1).uniform_(0, 1).tolist()[0]
            if prob <= self.change_prob:
                gold = self.conv2sliver(gold.data, shifted_preds.data, corpus_mask.data, seq_len, bat_size)


        # calculate gold energy on origin trellis
        gold_energy = self.calculate_energy(gold, trellis, mask)

        # average_batch
        loss = pred_energy - gold_energy
        # print("pred: {:.6f}, gold: {:.6f}, loss: {:.6f}".format(utils.to_scalar(pred_energy), utils.to_scalar(gold_energy), utils.to_scalar(loss)))
        if self.average_batch:
            loss = loss / bat_size
        return loss

    # ca
    def conv2sliver(self, gold, preds, corpus_mask, seq_len, bat_size):
        # unshift
        gold = gold.squeeze(2)
        preds = preds.squeeze(2)


        unshift_gold = gold % self.tagset_size
        unshift_preds = preds % self.tagset_size

        unshift_preds = unshift_preds[:-1]
        unshift_gold = unshift_gold[:-1]
        start_vec = preds.new(1, bat_size).long().fill_(self.start_tag)
        unshift_preds = torch.cat((start_vec, unshift_preds))
        unshift_gold = torch.cat((start_vec, unshift_gold))

        # find o in gold
        otag_mask = (unshift_gold == self.o_tag).long()

        # find corresponed pred
        pred_at_same_pos = (unshift_preds + 1)  * otag_mask - 1

        # check if the pred is out side corpus level tg
        book = {}
        for x, y in (corpus_mask[0].sum(2)==0).nonzero().tolist():
            if x not in book:
                book[x] = set([y])
            else:
                book[x] |= set([y])

        #print(book)
        rp_mask = pred_at_same_pos.new(pred_at_same_pos.size()).fill_(0)

        if book:
            row, col = pred_at_same_pos.size()
            
            for j in range(col):
                if j in book:
                    for i in range(row):
                        if pred_at_same_pos[i][j] in book[j]:
                            rp_mask[i][j] = 1

        # meger to new gold
        silver = unshift_preds * rp_mask + unshift_gold * (1 - rp_mask)
        # shift
        silver = self.shift_pred(silver[1:], seq_len, bat_size).contiguous()
        return silver


    def shift_pred(self, preds, seq_len, bat_size):
        start_vec = autograd.Variable(preds.new(1, bat_size).long().fill_(self.start_tag))
        end_vec = autograd.Variable(preds.new(1, bat_size).long().fill_(self.end_tag))
        preds = autograd.Variable(preds)
        shift_pred_tags = torch.cat((preds, end_vec))
        preds = torch.cat((start_vec, preds))
        preds = preds * self.tagset_size + shift_pred_tags
        preds = preds.unsqueeze(2)
        return preds

    def calculate_energy(self, preds, trellis, mask):
        seq_len = trellis.size(0)
        bat_size = trellis.size(1)

        energy = torch.gather(trellis.view(seq_len, bat_size, -1), 2, preds).view(seq_len, bat_size)
        energy = energy.masked_select(mask).sum()
        return energy

    def decode(self, trellis, mask):
        """Find the optimal path with viterbe decode

        args:
            trellis (size seq_len, bat_size, target_size_from, target_size_to) : crf trellis 
            mask (seq_len, bat_size) : mask for padding
        return:
            decoded sequence (size seq_len, bat_size)
        """
        # calculate batch size and seq len

        seq_len = trellis.size(0)
        bat_size = trellis.size(1)

        mask = 1 - mask
        decode_idx = trellis.new(seq_len-1, bat_size).long()

        # calculate forward score and checkpoint

        # build iter
        seq_iter = enumerate(trellis)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag

        forscores = inivalues[:, self.start_tag, :]  # bat_size * to_target_size
        back_points = list()

        # iter over last trellis
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + forscores.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)
            forscores, cur_bp = torch.max(cur_values, 1)
            cur_bp.masked_fill_(mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size), self.end_tag)

            back_points.append(cur_bp)

        pointer = back_points[-1][:, self.end_tag]
        decode_idx[-1] = pointer
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(bat_size, 1))
            decode_idx[idx] = pointer

        return decode_idx

    def build_cost_mask(self, trellis, target, cost_value):
        unshift_target = target % self.tagset_size
        cost_mask = trellis.new(trellis.size()).fill_(cost_value)

        # need a fancy way
        for row in range(len(cost_mask)):
            curr_tag = utils.to_scalar(unshift_target[row])
            cost_mask[row][:,curr_tag] = 0
            cost_mask[row][:,self.o_tag] += 0.5
            cost_mask[row][self.o_tag:] += 0.5
            cost_mask[row][self.o_tag, self.o_tag] -= 0.5
        return cost_mask

    def build_argumented_cost(self, trellis, target, cost_value):
        unshift_target = target % self.tagset_size
        argumented_cost = [self.build_cost_mask(trellis[0], target[0], 0.0)]
        for i in range(1, len(trellis)):
            cost = self.build_cost_mask(trellis[i], target[i], self.cost_value)
            argumented_cost.append(cost)
        return torch.stack(argumented_cost)
'''
class CRFLoss_sf(nn.Module):
    """loss for max_margin

    .. math::
        \sum_{j=1}^n \log (\phi(\hat{y}_{j-1}, \hat{y}_j, \mathbf{z}_j)) - \log (\sum_{\mathbf{y}' \in \mathbf{Y}(\mathbf{Z})} \prod_{j=1}^n \phi(y'_{j-1}, y'_j, \mathbf{z}_j) )

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch
        
    """

    def __init__(self, tag2idx, start_tag, end_tag, o_tag, cost_value=1.0, change_gold=False, change_prob=0.9, average_batch=True):
        super(CRFLoss_sf, self).__init__()
        self.tag2idx = tag2idx
        self.tagset_size = len(tag2idx)
        self.start_tag = start_tag
        self.o_tag = o_tag
        self.end_tag = end_tag
        self.average_batch = average_batch

        self.change_gold = change_gold
        self.change_prob = change_prob
        self.cost_value = cost_value
        print("SF, Penalization Cost: {:.2f}, Change Gold: {}, Prob: {:.2f}".format(self.cost_value, self.change_gold, self.change_prob))

    def forward(self, trellis, gold, mask, corpus_mask):
        """
        args:
            trellis (seq_len, bat_size, target_size_from, target_size_to) : crf trellis
            gold (seq_len, bat_size, 1) : golden state
            mask (size seq_len, bat_size) : mask for padding
        return:
            loss
        """
        # calculate batch size and seq len
        seq_len = trellis.size(0)
        bat_size = trellis.size(1)
        
        # calculate sentence score
        
        # predict on modified trellis
        if self.change_gold:
            preds = self.decode(trellis.data, mask.data)  
            shifted_preds = self.shift_pred(preds, seq_len, bat_size)
            prob = torch.Tensor(1).uniform_(0, 1).tolist()[0]
            if prob <= self.change_prob:
                gold = self.conv2sliver(gold.data, shifted_preds.data, corpus_mask.data, seq_len, bat_size)
        # calculate gold energy on origin trellis
        gold_energy = self.calculate_energy(gold, trellis, mask)
        
        argumented_cost = self.build_argumented_cost(trellis.data, gold, self.cost_value)
        trellis = trellis + autograd.Variable(argumented_cost)

        seq_iter = enumerate(trellis)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.start_tag, :]  # bat_size * to_target_size
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)
            cur_partition = utils.log_sum_exp(cur_values, self.tagset_size)
                  # (bat_size * from_target * to_target) -> (bat_size * to_target)
            partition = utils.switch(partition, cur_partition,
                                     mask[idx].contiguous().view(bat_size, 1).expand(bat_size, self.tagset_size)).contiguous().view(bat_size, -1)
            # the following two may achieve higher speed, but raise run-time error
            # new_partition = partition.clone()
            # new_partition.masked_scatter_(mask[idx].view(-1, 1).expand(bat_size, self.tagset_size), cur_partition)  #0 for partition, 1 for cur_partition
            # partition = new_partition
            
        #only need end at end_tag
        partition = partition[:, self.end_tag].sum()

        # average_batch
        loss = partition - gold_energy
        # print("pred: {:.6f}, gold: {:.6f}, loss: {:.6f}".format(utils.to_scalar(pred_energy), utils.to_scalar(gold_energy), utils.to_scalar(loss)))
        if self.average_batch:
            loss = loss / bat_size
        return loss

    # ca
    def conv2sliver(self, gold, preds, corpus_mask, seq_len, bat_size):
        # unshift
        gold = gold.squeeze(2)
        preds = preds.squeeze(2)


        unshift_gold = gold % self.tagset_size
        unshift_preds = preds % self.tagset_size

        unshift_preds = unshift_preds[:-1]
        unshift_gold = unshift_gold[:-1]
        start_vec = preds.new(1, bat_size).long().fill_(self.start_tag)
        unshift_preds = torch.cat((start_vec, unshift_preds))
        unshift_gold = torch.cat((start_vec, unshift_gold))

        # find o in gold
        otag_mask = (unshift_gold == self.o_tag).long()

        # find corresponed pred
        pred_at_same_pos = (unshift_preds + 1)  * otag_mask - 1

        # check if the pred is out side corpus level tg
        book = {}
        for x, y in (corpus_mask[0].sum(2)==0).nonzero().tolist():
            if x not in book:
                book[x] = set([y])
            else:
                book[x] |= set([y])

        #print(book)
        rp_mask = pred_at_same_pos.new(pred_at_same_pos.size()).fill_(0)

        if book:
            row, col = pred_at_same_pos.size()
            
            for j in range(col):
                if j in book:
                    for i in range(row):
                        if pred_at_same_pos[i][j] in book[j]:
                            rp_mask[i][j] = 1

        # meger to new gold
        silver = unshift_preds * rp_mask + unshift_gold * (1 - rp_mask)
        # shift
        silver = self.shift_pred(silver[1:], seq_len, bat_size).contiguous()
        return silver


    def shift_pred(self, preds, seq_len, bat_size):
        start_vec = autograd.Variable(preds.new(1, bat_size).long().fill_(self.start_tag))
        end_vec = autograd.Variable(preds.new(1, bat_size).long().fill_(self.end_tag))
        preds = autograd.Variable(preds)
        shift_pred_tags = torch.cat((preds, end_vec))
        preds = torch.cat((start_vec, preds))
        preds = preds * self.tagset_size + shift_pred_tags
        preds = preds.unsqueeze(2)
        return preds

    def calculate_energy(self, preds, trellis, mask):
        seq_len = trellis.size(0)
        bat_size = trellis.size(1)

        energy = torch.gather(trellis.view(seq_len, bat_size, -1), 2, preds).view(seq_len, bat_size)
        energy = energy.masked_select(mask).sum()
        return energy

    def decode(self, trellis, mask):
        """Find the optimal path with viterbe decode

        args:
            trellis (size seq_len, bat_size, target_size_from, target_size_to) : crf trellis 
            mask (seq_len, bat_size) : mask for padding
        return:
            decoded sequence (size seq_len, bat_size)
        """
        # calculate batch size and seq len

        seq_len = trellis.size(0)
        bat_size = trellis.size(1)

        mask = 1 - mask
        decode_idx = trellis.new(seq_len-1, bat_size).long()

        # calculate forward score and checkpoint

        # build iter
        seq_iter = enumerate(trellis)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag

        forscores = inivalues[:, self.start_tag, :]  # bat_size * to_target_size
        back_points = list()

        # iter over last trellis
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + forscores.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)
            forscores, cur_bp = torch.max(cur_values, 1)
            cur_bp.masked_fill_(mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size), self.end_tag)

            back_points.append(cur_bp)

        pointer = back_points[-1][:, self.end_tag]
        decode_idx[-1] = pointer
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(bat_size, 1))
            decode_idx[idx] = pointer

        return decode_idx

    def build_cost_mask(self, trellis, target, cost_value):
        unshift_target = target % self.tagset_size
        cost_mask = trellis.new(trellis.size()).fill_(0.0)

        # need a fancy way
        for row in range(len(cost_mask)):
            cost_mask[row][:,self.tag2idx["S-CHEMICAL"]] = cost_value
            cost_mask[row][:,self.tag2idx["B-CHEMICAL"]] = cost_value
            cost_mask[row][:,self.tag2idx["I-CHEMICAL"]] = cost_value
            cost_mask[row][:,self.tag2idx["E-CHEMICAL"]] = cost_value

            #cost_mask[row][:,self.tag2idx["S-GENE"]] = cost_value
            #cost_mask[row][:,self.tag2idx["B-GENE"]] = cost_value
            #cost_mask[row][:,self.tag2idx["I-GENE"]] = cost_value
            #cost_mask[row][:,self.tag2idx["E-GENE"]] = cost_value

        return cost_mask

    def build_argumented_cost(self, trellis, target, cost_value):
        unshift_target = target % self.tagset_size
        argumented_cost = [self.build_cost_mask(trellis[0], target[0], 0.0)]
        for i in range(1, len(trellis)):
            cost = self.build_cost_mask(trellis[i], target[i], self.cost_value)
            argumented_cost.append(cost)
        return torch.stack(argumented_cost)


class CRFLoss_rp(nn.Module):
    """loss for max_margin

    .. math::
        \sum_{j=1}^n \log (\phi(\hat{y}_{j-1}, \hat{y}_j, \mathbf{z}_j)) - \log (\sum_{\mathbf{y}' \in \mathbf{Y}(\mathbf{Z})} \prod_{j=1}^n \phi(y'_{j-1}, y'_j, \mathbf{z}_j) )

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch
        
    """

    def __init__(self, tag2idx, start_tag, end_tag, o_tag, change_prob=0.5, average_batch=True):
        super(CRFLoss_rp, self).__init__()
        self.tag2idx = tag2idx
        self.tagset_size = len(tag2idx)
        self.start_tag = start_tag
        self.o_tag = o_tag
        self.end_tag = end_tag
        self.average_batch = average_batch

        self.change_prob = change_prob
        print("RP, Prob: {:.2f}".format(self.change_prob))

    def forward(self, trellis, gold, mask, corpus_mask):
        """
        args:
            trellis (seq_len, bat_size, target_size_from, target_size_to) : crf trellis
            gold (seq_len, bat_size, 1) : golden state
            mask (size seq_len, bat_size) : mask for padding
        return:
            loss
        """
        # calculate batch size and seq len
        seq_len = trellis.size(0)
        bat_size = trellis.size(1)
        
        # calculate sentence score
        
        # predict on modified trellis
        prob = torch.Tensor(1).uniform_(0, 1).tolist()[0]
        if prob <= self.change_prob:
            #missing_mask = (corpus_mask != 1).float()
            preds = self.decode(trellis.data, mask.data)  
            shifted_preds = self.shift_pred(preds, seq_len, bat_size)
            gold = self.conv2sliver(gold.data, shifted_preds.data, corpus_mask, seq_len, bat_size)

                
        trellis = trellis #* corpus_mask
        gold_energy = self.calculate_energy(gold, trellis, mask)
        seq_iter = enumerate(trellis)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.start_tag, :]  # bat_size * to_target_size
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)
            cur_partition = utils.log_sum_exp(cur_values, self.tagset_size)
                  # (bat_size * from_target * to_target) -> (bat_size * to_target)
            partition = utils.switch(partition, cur_partition,
                                     mask[idx].contiguous().view(bat_size, 1).expand(bat_size, self.tagset_size)).contiguous().view(bat_size, -1)
            # the following two may achieve higher speed, but raise run-time error
            # new_partition = partition.clone()
            # new_partition.masked_scatter_(mask[idx].view(-1, 1).expand(bat_size, self.tagset_size), cur_partition)  #0 for partition, 1 for cur_partition
            # partition = new_partition
            
        #only need end at end_tag
        partition = partition[:, self.end_tag].sum()
        print(partition.size())
        # average_batch
        loss = partition - gold_energy
        # print("pred: {:.6f}, gold: {:.6f}, loss: {:.6f}".format(utils.to_scalar(pred_energy), utils.to_scalar(gold_energy), utils.to_scalar(loss)))
        if self.average_batch:
            loss = loss / bat_size
        return loss

    

    def shift_pred(self, preds, seq_len, bat_size):
        start_vec = autograd.Variable(preds.new(1, bat_size).long().fill_(self.start_tag))
        end_vec = autograd.Variable(preds.new(1, bat_size).long().fill_(self.end_tag))
        preds = autograd.Variable(preds)
        shift_pred_tags = torch.cat((preds, end_vec))
        preds = torch.cat((start_vec, preds))
        preds = preds * self.tagset_size + shift_pred_tags
        preds = preds.unsqueeze(2)
        return preds

    def calculate_energy(self, preds, trellis, mask):
        seq_len = trellis.size(0)
        bat_size = trellis.size(1)

        energy = torch.gather(trellis.view(seq_len, bat_size, -1), 2, preds).view(seq_len, bat_size)
        energy = energy.masked_select(mask).sum()
        return energy

    def decode(self, trellis, mask):
        """Find the optimal path with viterbe decode

        args:
            trellis (size seq_len, bat_size, target_size_from, target_size_to) : crf trellis 
            mask (seq_len, bat_size) : mask for padding
        return:
            decoded sequence (size seq_len, bat_size)
        """
        # calculate batch size and seq len

        seq_len = trellis.size(0)
        bat_size = trellis.size(1)

        mask = 1 - mask
        decode_idx = trellis.new(seq_len-1, bat_size).long()

        # calculate forward score and checkpoint

        # build iter
        seq_iter = enumerate(trellis)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag

        forscores = inivalues[:, self.start_tag, :]  # bat_size * to_target_size
        back_points = list()

        # iter over last trellis
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + forscores.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)
            forscores, cur_bp = torch.max(cur_values, 1)
            cur_bp.masked_fill_(mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size), self.end_tag)

            back_points.append(cur_bp)

        pointer = back_points[-1][:, self.end_tag]
        decode_idx[-1] = pointer
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(bat_size, 1))
            decode_idx[idx] = pointer

        return decode_idx

    def conv2sliver(self, gold, preds, corpus_mask, seq_len, bat_size):
        # unshift
        gold = gold.squeeze(2)
        preds = preds.squeeze(2)


        unshift_gold = gold % self.tagset_size
        unshift_preds = preds % self.tagset_size

        unshift_preds = unshift_preds[:-1]
        unshift_gold = unshift_gold[:-1]
        start_vec = preds.new(1, bat_size).long().fill_(self.start_tag)
        unshift_preds = torch.cat((start_vec, unshift_preds))
        unshift_gold = torch.cat((start_vec, unshift_gold))

        # find o in gold
        otag_mask = (unshift_gold == self.o_tag).long()

        # find corresponed pred
        pred_at_same_pos = (unshift_preds + 1)  * otag_mask - 1

        # check if the pred is out side corpus level tg
        book = {}
        for x, y in (corpus_mask[0].sum(2)==0).nonzero().tolist():
            if x not in book:
                book[x] = set([y])
            else:
                book[x] |= set([y])

        #print(book)
        rp_mask = pred_at_same_pos.new(pred_at_same_pos.size()).fill_(0)

        if book:
            row, col = pred_at_same_pos.size()
            
            for j in range(col):
                if j in book:
                    for i in range(row):
                        if pred_at_same_pos[i][j] in book[j]:
                            rp_mask[i][j] = 1

        # meger to new gold
        silver = unshift_preds * rp_mask + unshift_gold * (1 - rp_mask)
        # shift
        silver = self.shift_pred(silver[1:], seq_len, bat_size).contiguous()
        return silver




class CRFLoss_mm(nn.Module):
    """loss for max_margin

    .. math::
        \sum_{j=1}^n \log (\phi(\hat{y}_{j-1}, \hat{y}_j, \mathbf{z}_j)) - \log (\sum_{\mathbf{y}' \in \mathbf{Y}(\mathbf{Z})} \prod_{j=1}^n \phi(y'_{j-1}, y'_j, \mathbf{z}_j) )

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch
        
    """

    def __init__(self, tagset_size, start_tag, end_tag, o_tag, cost_value=1.0, change_gold=False, change_prob=0.9, average_batch=True):
        super(CRFLoss_mm, self).__init__()
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.o_tag = o_tag
        self.end_tag = end_tag
        self.average_batch = average_batch
        self.cost_value = cost_value
        self.change_gold = change_gold
        self.change_prob = change_prob
        
        print("Cost Value {:.2f}, Change Gold: {}, Prob: {:.2f}".format(self.cost_value, self.change_gold, self.change_prob))

    def forward(self, trellis, gold, mask, corpus_mask):
        """
        args:
            trellis (seq_len, bat_size, target_size_from, target_size_to) : crf trellis
            gold (seq_len, bat_size, 1) : golden state
            mask (size seq_len, bat_size) : mask for padding
        return:
            loss
        """
        # calculate batch size and seq len
        seq_len = trellis.size(0)
        bat_size = trellis.size(1)
        
        # calculate sentence score
        
        argumented_cost = self.build_argumented_cost(trellis.data, gold, self.cost_value)
        trellis = trellis + autograd.Variable(argumented_cost)
        
        # predict on modified trellis

        preds = self.decode(trellis.data, mask.data)  
        shifted_preds = self.shift_pred(preds, seq_len, bat_size)

        pred_energy = self.calculate_energy(shifted_preds, trellis, mask)

        if self.change_gold:
            prob = torch.Tensor(1).uniform_(0, 1).tolist()[0]
            if prob <= self.change_prob:
                gold = self.conv2sliver(gold.data, shifted_preds.data, corpus_mask.data, seq_len, bat_size)


        # calculate gold energy on origin trellis
        gold_energy = self.calculate_energy(gold, trellis, mask)

        # average_batch
        loss = pred_energy - gold_energy
        # print("pred: {:.6f}, gold: {:.6f}, loss: {:.6f}".format(utils.to_scalar(pred_energy), utils.to_scalar(gold_energy), utils.to_scalar(loss)))
        if self.average_batch:
            loss = loss / bat_size
        return loss

    # ca
    def conv2sliver(self, gold, preds, corpus_mask, seq_len, bat_size):
        # unshift
        gold = gold.squeeze(2)
        preds = preds.squeeze(2)


        unshift_gold = gold % self.tagset_size
        unshift_preds = preds % self.tagset_size

        unshift_preds = unshift_preds[:-1]
        unshift_gold = unshift_gold[:-1]
        start_vec = preds.new(1, bat_size).long().fill_(self.start_tag)
        unshift_preds = torch.cat((start_vec, unshift_preds))
        unshift_gold = torch.cat((start_vec, unshift_gold))

        # find o in gold
        otag_mask = (unshift_gold == self.o_tag).long()

        # find corresponed pred
        pred_at_same_pos = (unshift_preds + 1)  * otag_mask - 1

        # check if the pred is out side corpus level tg
        book = {}
        for x, y in (corpus_mask[0].sum(2)==0).nonzero().tolist():
            if x not in book:
                book[x] = set([y])
            else:
                book[x] |= set([y])

        #print(book)
        rp_mask = pred_at_same_pos.new(pred_at_same_pos.size()).fill_(0)

        if book:
            row, col = pred_at_same_pos.size()
            
            for j in range(col):
                if j in book:
                    for i in range(row):
                        if pred_at_same_pos[i][j] in book[j]:
                            rp_mask[i][j] = 1

        # meger to new gold
        silver = unshift_preds * rp_mask + unshift_gold * (1 - rp_mask)
        # shift
        silver = self.shift_pred(silver[1:], seq_len, bat_size).contiguous()
        return silver


    def shift_pred(self, preds, seq_len, bat_size):
        start_vec = autograd.Variable(preds.new(1, bat_size).long().fill_(self.start_tag))
        end_vec = autograd.Variable(preds.new(1, bat_size).long().fill_(self.end_tag))
        preds = autograd.Variable(preds)
        shift_pred_tags = torch.cat((preds, end_vec))
        preds = torch.cat((start_vec, preds))
        preds = preds * self.tagset_size + shift_pred_tags
        preds = preds.unsqueeze(2)
        return preds

    def calculate_energy(self, preds, trellis, mask):
        seq_len = trellis.size(0)
        bat_size = trellis.size(1)

        energy = torch.gather(trellis.view(seq_len, bat_size, -1), 2, preds).view(seq_len, bat_size)
        energy = energy.masked_select(mask).sum()
        return energy

    def decode(self, trellis, mask):
        """Find the optimal path with viterbe decode

        args:
            trellis (size seq_len, bat_size, target_size_from, target_size_to) : crf trellis 
            mask (seq_len, bat_size) : mask for padding
        return:
            decoded sequence (size seq_len, bat_size)
        """
        # calculate batch size and seq len

        seq_len = trellis.size(0)
        bat_size = trellis.size(1)

        mask = 1 - mask
        decode_idx = trellis.new(seq_len-1, bat_size).long()

        # calculate forward score and checkpoint

        # build iter
        seq_iter = enumerate(trellis)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag

        forscores = inivalues[:, self.start_tag, :]  # bat_size * to_target_size
        back_points = list()

        # iter over last trellis
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + forscores.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)
            forscores, cur_bp = torch.max(cur_values, 1)
            cur_bp.masked_fill_(mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size), self.end_tag)

            back_points.append(cur_bp)

        pointer = back_points[-1][:, self.end_tag]
        decode_idx[-1] = pointer
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(bat_size, 1))
            decode_idx[idx] = pointer

        return decode_idx

    def build_cost_mask(self, trellis, target, cost_value):
        unshift_target = target % self.tagset_size
        cost_mask = trellis.new(trellis.size()).fill_(cost_value)

        # need a fancy way
        for row in range(len(cost_mask)):
            curr_tag = utils.to_scalar(unshift_target[row])
            cost_mask[row][:,curr_tag] = 0
        return cost_mask

    def build_argumented_cost(self, trellis, target, cost_value):
        unshift_target = target % self.tagset_size
        argumented_cost = [self.build_cost_mask(trellis[0], target[0], 0.0)]
        for i in range(1, len(trellis)):
            cost = self.build_cost_mask(trellis[i], target[i], self.cost_value)
            argumented_cost.append(cost)
        return torch.stack(argumented_cost)

class CRFDecode_vb():
    """Batch-mode viterbi decode

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch
        
    """

    def __init__(self, tagset_size, start_tag, end_tag, average_batch=True):
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.average_batch = average_batch


    def decode(self, scores, mask):
        """Find the optimal path with viterbe decode

        args:
            scores (size seq_len, bat_size, target_size_from, target_size_to) : crf scores 
            mask (seq_len, bat_size) : mask for padding
        return:
            decoded sequence (size seq_len, bat_size)
        """
        # calculate batch size and seq len

        seq_len = scores.size(0)
        bat_size = scores.size(1)

        mask = 1 - mask
        #decode_idx = scores.new(seq_len-1, bat_size).long()
        decode_idx = torch.LongTensor(seq_len-1, bat_size)

        # calculate forward score and checkpoint

        # build iter
        seq_iter = enumerate(scores)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        forscores = inivalues[:, self.start_tag, :]  # bat_size * to_target_size
        back_points = list()
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + forscores.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)

            forscores, cur_bp = torch.max(cur_values, 1)
            
            cur_bp.masked_fill_(mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size), self.end_tag)

            back_points.append(cur_bp)

        pointer = back_points[-1][:, self.end_tag]
        decode_idx[-1] = pointer
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(bat_size, 1))
            decode_idx[idx] = pointer
        return decode_idx