"""
.. module:: predictor
    :synopsis: prediction method (for un-annotated text)
 
.. moduleauthor:: Liyuan Liu
"""

import torch
import torch.autograd as autograd
import numpy as np
import itertools
import sys
from tqdm import tqdm

from model.crf import CRFDecode_vb
from model.utils import *
import model.utils as utils
import functools
from tqdm import tqdm
import sys

class predict:
    """Base class for prediction, provide method to calculate f1 score and accuracy 

    args: 
        if_cuda: if use cuda to speed up 
        l_map: dictionary for labels 
        label_seq: type of decode function, set `True` to couple label with text, or set 'False' to insert label into test
        batch_size: size of batch in decoding
    """

    def __init__(self, if_cuda, l_map, label_seq = True, batch_size = 50):
        self.if_cuda = if_cuda
        self.l_map = l_map
        self.r_l_map = revlut(l_map)
        self.batch_size = batch_size
        if label_seq:
            self.decode_str = self.decode_l
        else:
            self.decode_str = self.decode_s

    def decode_l(self, feature, label):
        """
        decode a sentence coupled with label

        args:
            feature (list): words list
            label (list): label list
        """
        return '\n'.join(map(lambda t: t[0] + ' '+ self.r_l_map[t[1]], zip(feature, label)))

    def decode_s(self, feature, label):
        """
        decode a sentence in the format of <>

        args:
            feature (list): words list
            label (list): label list
        """
        chunks = ""
        current = None

        for f, y in zip(feature, label):
            label = self.r_l_map[y]

            if label.startswith('B-'):

                if current is not None:
                    chunks += "</"+current+"> "
                current = label[2:]
                chunks += "<"+current+"> " + f + " "

            elif label.startswith('S-'):

                if current is not None:
                    chunks += " </"+current+"> "
                current = label[2:]
                chunks += "<"+current+"> " + f + " </"+current+"> "
                current = None

            elif label.startswith('I-'):

                if current is not None:
                    base = label[2:]
                    if base == current:
                        chunks += f+" "
                    else:
                        chunks += "</"+current+"> <"+base+"> " + f + " "
                        current = base
                else:
                    current = label[2:]
                    chunks += "<"+current+"> " + f + " "

            elif label.startswith('E-'):

                if current is not None:
                    base = label[2:]
                    if base == current:
                        chunks += f + " </"+base+"> "
                        current = None
                    else:
                        chunks += "</"+current+"> <"+base+"> " + f + " </"+base+"> "
                        current = None

                else:
                    current = label[2:]
                    chunks += "<"+current+"> " + f + " </"+current+"> "
                    current = None

            else:
                if current is not None:
                    chunks += "</"+current+"> "
                chunks += f+" "
                current = None

        if current is not None:
            chunks += "</"+current+"> "

        return chunks

    def output_batch(self, ner_model, documents, fout, file_no):
        """
        decode the whole corpus in the specific format by calling apply_model to fit specific models

        args:
            ner_model: sequence labeling model
            feature (list): list of words list
            fout: output file
        """
        ner_model.eval()

        d_len = len(documents)
        for d_ind in tqdm( range(0, d_len), mininterval=1,
                desc=' - Process', leave=False, file=sys.stdout):
            fout.write('-DOCSTART- -DOCSTART- -DOCSTART-\n\n')
            features = documents[d_ind]
            f_len = len(features)
            for ind in range(0, f_len, self.batch_size):
                eind = min(f_len, ind + self.batch_size)
                labels = self.apply_model(ner_model, features[ind: eind], file_no)
                labels = torch.unbind(labels, 1)

                for ind2 in range(ind, eind):
                    f = features[ind2]
                    l = labels[ind2 - ind][0: len(f) ]
                    fout.write(self.decode_str(features[ind2], l) + '\n\n')

    def apply_model(self, ner_model, features):
        """
        template function for apply_model

        args:
            ner_model: sequence labeling model
            feature (list): list of words list
        """
        return None

class predict_w(predict):
    """prediction class for word level model (LSTM-CRF)

    args: 
        if_cuda: if use cuda to speed up 
        f_map: dictionary for words
        l_map: dictionary for labels
        pad_word: word padding
        pad_label: label padding
        start_label: start label 
        label_seq: type of decode function, set `True` to couple label with text, or set 'False' to insert label into test
        batch_size: size of batch in decoding
        caseless: caseless or not
    """
   
    def __init__(self, if_cuda, f_map, l_map, pad_word, pad_label, start_label, label_seq = True, batch_size = 50, caseless=True):
        predict.__init__(self, if_cuda, l_map, label_seq, batch_size)
        self.decoder = CRFDecode_vb(len(l_map), start_label, pad_label)
        self.pad_word = pad_word
        self.f_map = f_map
        self.l_map = l_map
        self.caseless = caseless
        
    def apply_model(self, ner_model, features):
        """
        apply_model function for LSTM-CRF

        args:
            ner_model: sequence labeling model
            feature (list): list of words list
        """
        if self.caseless:
            features = list(map(lambda t: list(map(lambda x: x.lower(), t)), features))
        features = encode_safe(features, self.f_map, self.f_map['<unk>'])
        f_len = max(map(lambda t: len(t) + 1, features))

        masks = torch.ByteTensor(list(map(lambda t: [1] * (len(t) + 1) + [0] * (f_len - len(t) - 1), features)))
        word_features = torch.LongTensor(list(map(lambda t: t + [self.pad_word] * (f_len - len(t)), features)))

        if self.if_cuda:
            fea_v = autograd.Variable(word_features.transpose(0, 1)).cuda()
            mask_v = masks.transpose(0, 1).cuda()
        else:
            fea_v = autograd.Variable(word_features.transpose(0, 1))
            mask_v = masks.transpose(0, 1).contiguous()

        scores, _ = ner_model(fea_v)
        decoded = self.decoder.decode(scores.data, mask_v)

        return decoded

class predict_wc(predict):
    """prediction class for LM-LSTM-CRF

    args: 
        if_cuda: if use cuda to speed up 
        f_map: dictionary for words
        c_map: dictionary for chars
        l_map: dictionary for labels
        pad_word: word padding
        pad_char: word padding
        pad_label: label padding
        start_label: start label 
        label_seq: type of decode function, set `True` to couple label with text, or set 'False' to insert label into test
        batch_size: size of batch in decoding
        caseless: caseless or not
    """
   
    def __init__(self, if_cuda, f_map, c_map, l_map, pad_word, pad_char, pad_label, start_label, label_seq = True, batch_size = 50, caseless=True):
        predict.__init__(self, if_cuda, l_map, label_seq, batch_size)
        self.decoder = CRFDecode_vb(len(l_map), start_label, pad_label)
        self.pad_word = pad_word
        self.pad_char = pad_char
        self.f_map = f_map
        self.c_map = c_map
        self.l_map = l_map
        self.caseless = caseless
        
    def calc_score(self, ner_model, features, file_no):
        """
        apply_model function for LM-LSTM-CRF

        args:
            ner_model: sequence labeling model
            feature (list): list of words list
        """
        char_features = encode2char_safe(features, self.c_map)

        if self.caseless:
            word_features = encode_safe(list(map(lambda t: list(map(lambda x: x.lower(), t)), features)), self.f_map, self.f_map['<unk>'])
        else:
            word_features = encode_safe(features, self.f_map, self.f_map['<unk>'])

        fea_len = [list( map( lambda t: len(t) + 1, f) ) for f in char_features]
        forw_features = concatChar(char_features, self.c_map)

        word_len = max(map(lambda t: len(t) + 1, word_features))
        char_len = max(map(lambda t: len(t[0]) + word_len - len(t[1]), zip(forw_features, word_features)))
        forw_t = list( map( lambda t: t + [self.pad_char] * ( char_len - len(t) ), forw_features ) )
        back_t = torch.LongTensor( list( map( lambda t: t[::-1], forw_t ) ) )
        forw_t = torch.LongTensor( forw_t )
        forw_p = torch.LongTensor( list( map( lambda t: list(itertools.accumulate( t + [1] * (word_len - len(t) ) ) ), fea_len) ) )
        back_p = torch.LongTensor( list( map( lambda t: [char_len - 1] + [ char_len - 1 - tup for tup in t[:-1] ], forw_p) ) )

        masks = torch.ByteTensor(list(map(lambda t: [1] * (len(t) + 1) + [0] * (word_len - len(t) - 1), word_features)))
        word_t = torch.LongTensor(list(map(lambda t: t + [self.pad_word] * (word_len - len(t)), word_features)))

        if self.if_cuda:
            f_f = autograd.Variable(forw_t.transpose(0, 1)).cuda()
            f_p = autograd.Variable(forw_p.transpose(0, 1)).cuda()
            b_f = autograd.Variable(back_t.transpose(0, 1)).cuda()
            b_p = autograd.Variable(back_p.transpose(0, 1)).cuda()
            w_f = autograd.Variable(word_t.transpose(0, 1)).cuda()
            mask_v = masks.transpose(0, 1).cuda()
        else:
            f_f = autograd.Variable(forw_t.transpose(0, 1))
            f_p = autograd.Variable(forw_p.transpose(0, 1))
            b_f = autograd.Variable(back_t.transpose(0, 1))
            b_p = autograd.Variable(back_p.transpose(0, 1))
            w_f = autograd.Variable(word_t.transpose(0, 1))
            mask_v = masks.transpose(0, 1)

        scores = ner_model(f_f, f_p, b_f, b_p, w_f, file_no)
        decoded = self.decoder.decode(scores.data, mask_v)

        return decoded

    def apply_model(self, ner_model, dataset_loader, crf_no, crit_ner, verbose=0):
        """
        calculate score for pre-selected metrics

        args: 
            ner_model: LM-LSTM-CRF model
            dataset_loader: loader class for test set
        """
        ner_model.eval()

        for f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v, corpus_mask_v, reorder, pos_v, chunk_v, lexicon_v in itertools.chain.from_iterable(dataset_loader):
            f_f, f_p, b_f, b_p, w_f, _, mask_v, corpus_mask_v, pos_v, chunk_v, lexicon_v = self.packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v, corpus_mask_v, pos_v, chunk_v, lexicon_v)
            scores = ner_model(f_f, f_p, b_f, b_p, w_f, crf_no, corpus_mask_v, pos_v, chunk_v, lexicon_v)

            decoded = self.decoder.decode(scores.data, mask_v.data)
        return decoded




class Predictor:
    """Base class for prediction, provide method to calculate f1 score and accuracy 

    args: 
        if_cuda: if use cuda to speed up 
        l_map: dictionary for labels 
        label_seq: type of decode function, set `True` to couple label with text, or set 'False' to insert label into test
        batch_size: size of batch in decoding
    """

    def __init__(self, l_map, packer, label_seq = True, batch_size = 50):
        self.l_map = l_map
        self.decoder = CRFDecode_vb(len(l_map), l_map['<start>'], l_map['<pad>'])
        self.packer = packer
        self.r_l_map = revlut(l_map)
        self.batch_size = batch_size
        if label_seq:
            self.decode_str = self.decode_l
        else:
            self.decode_str = self.decode_s

    def decode_l(self, feature, label):
        """
        decode a sentence coupled with label

        args:
            feature (list): words list
            label (list): label list
        """
        return '\n'.join(map(lambda t: t[0] + ' '+ self.r_l_map[t[1]], zip(feature, label)))

    def decode_s(self, feature, label):
        """
        decode a sentence in the format of <>

        args:
            feature (list): words list
            label (list): label list
        """
        chunks = ""
        current = None

        for f, y in zip(feature, label):
            label = self.r_l_map[y]

            if label.startswith('B-'):

                if current is not None:
                    chunks += "</"+current+"> "
                current = label[2:]
                chunks += "<"+current+"> " + f + " "

            elif label.startswith('S-'):

                if current is not None:
                    chunks += " </"+current+"> "
                current = label[2:]
                chunks += "<"+current+"> " + f + " </"+current+"> "
                current = None

            elif label.startswith('I-'):

                if current is not None:
                    base = label[2:]
                    if base == current:
                        chunks += f+" "
                    else:
                        chunks += "</"+current+"> <"+base+"> " + f + " "
                        current = base
                else:
                    current = label[2:]
                    chunks += "<"+current+"> " + f + " "

            elif label.startswith('E-'):

                if current is not None:
                    base = label[2:]
                    if base == current:
                        chunks += f + " </"+base+"> "
                        current = None
                    else:
                        chunks += "</"+current+"> <"+base+"> " + f + " </"+base+"> "
                        current = None

                else:
                    current = label[2:]
                    chunks += "<"+current+"> " + f + " </"+current+"> "
                    current = None

            else:
                if current is not None:
                    chunks += "</"+current+"> "
                chunks += f+" "
                current = None

        if current is not None:
            chunks += "</"+current+"> "

        return chunks

    def output_batch(self, ner_model, documents, fout, crf_no):
        """
        decode the whole corpus in the specific format by calling apply_model to fit specific models

        args:
            ner_model: sequence labeling model
            feature (list): list of words list
            fout: output file
        """
        ner_model.eval()

        d_len = len(documents)
        for d_ind in tqdm( range(0, d_len), mininterval=1,
                desc=' - Process', leave=False, file=sys.stdout):
            fout.write('-DOCSTART- -DOCSTART- -DOCSTART-\n\n')
            features = documents[d_ind]
            f_len = len(features)
            for ind in range(0, f_len, self.batch_size):
                eind = min(f_len, ind + self.batch_size)
                labels = self.apply_model(ner_model, features[ind: eind], file_no)
                labels = torch.unbind(labels, 1)

                for ind2 in range(ind, eind):
                    f = features[ind2]
                    l = labels[ind2 - ind][0: len(f) ]
                    fout.write(self.decode_str(features[ind2], l) + '\n\n')

    def predict(self, ner_model, dataset_loader, crf_no, pred_method, merge_batch=False, totag=False):
        """
        calculate score for pre-selected metrics

        args: 
            ner_model: LM-LSTM-CRF model
            dataset_loader: loader class for test set
        """
        ner_model.eval()
        corpus_labels = []
        num_sample = sum(map(lambda t: len(t), dataset_loader)) 
        for f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v, corpus_mask_v, reorder in tqdm(
            itertools.chain.from_iterable(dataset_loader), mininterval=2,
            desc=' - Total it %d' % (num_sample), leave=False, file=sys.stdout):
            f_f, f_p, b_f, b_p, w_f, _, mask_v, corpus_mask_v = self.packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v, corpus_mask_v, volatile=True)
            labels, scores = self.predict_batch(ner_model, crf_no, f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v, corpus_mask_v, pred_method)
            
            labels = torch.unbind(labels, 1)
            _, length = torch.unbind(len_v, 1)
            length = length.tolist()

            unpad_labels = []
            for pad_labels, sent_len, rerdr in zip(labels, length, reorder):
                # subtract the <start> tag
                unpad_labels.append((rerdr, pad_labels[:sent_len - 1]))
            if totag:
                unpad_labels = [(rerdr, [self.r_l_map[idx_label] for idx_label in sent_idx_label]) for rerdr, sent_idx_label in unpad_labels]
            corpus_labels.append(unpad_labels)
        if merge_batch:
            corpus_labels = functools.reduce(lambda x, y: x + y, corpus_labels)
        return corpus_labels

    def predict_batch(self, ner_model, crf_no, f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v, corpus_mask_v, pred_method):
        """
        calculate score for pre-selected metrics

        args: 
            ner_model: LM-LSTM-CRF model
            dataset_loader: loader class for test set
        """
        if ner_model.training:
            ner_model.eval()
        scores = ner_model(f_f, f_p, b_f, b_p, w_f, crf_no, corpus_mask_v)
        
        assert pred_method in ["M", "U"]
        
        if pred_method == "M":
            # no matter take sigmoid or not, setting undesired scores to -inf
            neg_inf_scores = autograd.Variable(torch.FloatTensor(np.full(scores.shape, -1e9))).cuda()
            selected_scores = utils.switch(neg_inf_scores.contiguous(), scores.contiguous(), corpus_mask_v).view(scores.shape)
            decoded = self.decoder.decode(selected_scores.data, mask_v.data)
            return decoded, scores
        
        if pred_method == "U":
            decoded = self.decoder.decode(scores.data, mask_v.data)
            for i in range(decoded.shape[0]):
                for j in range(decoded.shape[1]):
                    idx_annotated = np.where(corpus_mask_v[i,j,0].cpu().data)[0]
                    if not decoded[i,j] in idx_annotated:
                        decoded[i,j] = self.l_map['O']
            return decoded, scores

