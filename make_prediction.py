import torch, argparse, json, pickle, itertools, os, time
from copy import deepcopy
from model.lm_lstm_crf import *
from model.data_util import *
from model.crf import *
from pathlib import Path


def load_data(train_features, train_labels, train_args):
    crf2train_dataloader = build_crf2dataloader(train_args['crf2corpus'], train_features, train_labels, train_args['batch_size'], train_args['corpus_missing_tagspace'], train_args['corpus_mask_value'], train_args['tag2idx'], train_args['chr2idx'], train_args['token2idx'], train_args['caseless'], shuffle=False, drop_last=False) 
    return crf2train_dataloader

# update scores for supervised viterbi decoding
def update_s(s, l, idx_annotated, O_idx):
    new_s = deepcopy(s)
    if l != O_idx:
        # if the label is not "O", set prob of every other NE to 0
        new_s[:l, :] = -np.Inf
        new_s[(l+1):, :] = -np.Inf
    else:
        # if the label is "O", set prob of every annotated NE to 0
        new_s[idx_annotated, :] = -np.Inf
    
    return new_s

def make_silver(dataloader):
    start_time = time.time()
    iii=0
    new_dataloader = [[], [], [], []]    
    for f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v, corpus_mask_v, reorder in itertools.chain.from_iterable(dataloader):
        f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, corpus_mask_v = packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v, corpus_mask_v)
        tg_v = tg_v.cpu()
        corpus_mask_v = corpus_mask_v.cpu()
        mask_v = mask_v.cpu()
        
        scores = ner_model(f_f.cpu(), f_p.cpu(), b_f.cpu(), b_p.cpu(), w_f.cpu(), 0, corpus_mask_v.cpu())
        pred = decoder.decode(scores.data, mask_v.data, negated = False)
        
        seq_len = scores.size(0)
        bat_size = scores.size(1)
        tagset_size = len(train_args['tag2idx'])
        
        gold_labels = (tg_v / 35).view(tg_v.shape[0], tg_v.shape[1]).data.numpy()
        
        bat_full_probas = []
        
        for i in range(bat_size):
            curr_seq = scores[:,i,:,:]
            curr_mask = mask_v[:,i]
            cliped_seq = curr_seq[:sum(curr_mask.data)]
            fore_probas = [cliped_seq[0, train_args['tag2idx']['<start>'], :]] + [0 for r in range(len(cliped_seq)-2)]
            back_probas = [0 for r in range(len(cliped_seq)-2)] + [cliped_seq[-1, :, train_args['tag2idx']['<pad>']]]
            
            for j in range(1, cliped_seq.shape[0]-1):
                prev_proba = fore_probas[j-1]
                curr_score = cliped_seq[j]
                curr_score = prev_proba.view(tagset_size,1).expand(tagset_size,tagset_size) + curr_score
                curr_proba = utils.log_sum_exp(curr_score.view(1,tagset_size,tagset_size), tagset_size)
                fore_probas[j] = curr_proba.view(-1)
            
            for j in range(cliped_seq.shape[0]-3, -1, -1):
                next_proba = back_probas[j+1]
                next_score = cliped_seq[j+1]
                next_score = next_proba.expand(tagset_size,tagset_size) + next_score
                curr_proba = utils.log_sum_exp(next_score.transpose(0,1).contiguous().view(1,tagset_size,tagset_size), tagset_size)
                back_probas[j] = curr_proba.view(-1)
            
            full_probas = []
            for j in range(len(fore_probas)):
                full_proba = (fore_probas[j] + back_probas[j]).data.numpy()
                full_proba = full_proba - np.mean(full_proba)
                full_proba = np.e ** full_proba
                full_proba = full_proba / np.sum(full_proba)
                full_probas.append(list(full_proba))
            
            bat_full_probas.append(full_probas)
            
        silver1 = deepcopy(gold_labels)
        silver2 = deepcopy(gold_labels)
        silver3 = deepcopy(gold_labels)
        for i in range(1,gold_labels.shape[0]):
            for j in range(gold_labels.shape[1]):
                l = gold_labels[i,j]
                if l == 6:
                    break
                p = pred[i-1,j]
                proba = bat_full_probas[j][i-1][p]
                idx_annotated = np.where(corpus_mask_v[i,j,0].data)[0]
                idx_annotated = np.array([r for r in idx_annotated if r!=0])
                if l == 0 and p != 0 and not p in idx_annotated:
                    if proba > 0.7:
                        silver1[i,j] = p
                    if proba > 0.9:
                        silver2[i,j] = p
                    if proba > 0.5:
                        silver3[i,j] = p
        
        scores = scores.data.numpy()
        
        new_scores = []
        for i in range(scores.shape[0]):
            new_scores_1 = []
            for j in range(scores.shape[1]):
                s, l = scores[i,j,:,:], gold_labels[i,j]
                idx_annotated = np.where(corpus_mask_v[i,j,0].data)[0]
                idx_annotated = np.array([r for r in idx_annotated if r!=0])
                new_scores_1.append(update_s(s, l, idx_annotated, int(train_args['tag2idx']['O'])))
            new_scores.append(new_scores_1)
        
        new_scores = torch.from_numpy(np.array(new_scores))
        supervised_pred = decoder.decode(new_scores, mask_v.data, negated = False)
        start_label = train_args['tag2idx']['<start>']
        pad_label = train_args['tag2idx']['<pad>']
        silver4 = np.array([[start_label for r in range(supervised_pred.shape[1])]] + supervised_pred.numpy().tolist())
        
        tg_v_1, tg_v_2, tg_v_3, tg_v_4 = deepcopy(tg_v.data.numpy()), deepcopy(tg_v.data.numpy()), deepcopy(tg_v.data.numpy()), deepcopy(tg_v.data.numpy())
        for i in range(bat_size):
            curr_mask = mask_v[:,i]
            label_size = len(train_args['tag2idx'])
            cur_len = sum(curr_mask).data.numpy()[0]
            threshold = seq_len
            
            i_l = silver1[:,i]
            curr_tg_v = [i_l[ind] * label_size + i_l[ind + 1] for ind in range(0, threshold-1)] + [i_l[cur_len-1] * label_size + pad_label]
            tg_v_1[:,i] = np.array(curr_tg_v).reshape(-1,1)
            
            i_l = silver2[:,i]
            curr_tg_v = [i_l[ind] * label_size + i_l[ind + 1] for ind in range(0, threshold-1)] + [i_l[cur_len-1] * label_size + pad_label]
            tg_v_2[:,i] = np.array(curr_tg_v).reshape(-1,1)
            
            i_l = silver3[:,i]
            curr_tg_v = [i_l[ind] * label_size + i_l[ind + 1] for ind in range(0, threshold-1)] + [i_l[cur_len-1] * label_size + pad_label]
            tg_v_3[:,i] = np.array(curr_tg_v).reshape(-1,1)
            
            i_l = silver4[:,i]
            curr_tg_v = [i_l[ind] * label_size + i_l[ind + 1] for ind in range(0, threshold-1)] + [i_l[cur_len-1] * label_size + pad_label]
            tg_v_4[:,i] = np.array(curr_tg_v).reshape(-1,1)
            
        tg_v_1, tg_v_2, tg_v_3, tg_v_4 = torch.from_numpy(tg_v_1), torch.from_numpy(tg_v_2), torch.from_numpy(tg_v_3), torch.from_numpy(tg_v_4)
        
        corpus_mask_v = corpus_mask_v.data.numpy()
        corpus_mask_v[:,:,:] = 1
        corpus_mask_v = torch.from_numpy(corpus_mask_v)
        
        f_f, f_p, b_f, b_p, w_f = f_f.cpu(), f_p.cpu(), b_f.cpu(), b_p.cpu(), w_f.cpu()
        new_dataloader[0].append([f_f, f_p, b_f, b_p, w_f, tg_v_1, mask_v, len_v, corpus_mask_v, reorder])
        new_dataloader[1].append([f_f, f_p, b_f, b_p, w_f, tg_v_2, mask_v, len_v, corpus_mask_v, reorder])
        new_dataloader[2].append([f_f, f_p, b_f, b_p, w_f, tg_v_3, mask_v, len_v, corpus_mask_v, reorder])
        new_dataloader[3].append([f_f, f_p, b_f, b_p, w_f, tg_v_4, mask_v, len_v, corpus_mask_v, reorder])
        iii+=1
        if iii%1000==0:
            print(iii)
            print(time.time()-start_time)
    
    return new_dataloader



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make prediction with pretrained models')
    parser.add_argument('--checkpoint', help='checkpoint to be loaded')
    parser.add_argument('--train_args', help='args to be loaded')
    parser.add_argument('--train_file', nargs='+', default=['./corpus/train/BC2GM-IOBES/train.tsv',
                                                            './corpus/train/BC4CHEMD-IOBES/train.tsv',
                                                            './corpus/train/BC5CDR-IOBES/train.tsv',
                                                            './corpus/train/JNLPBA-IOBES/train.tsv',
                                                            './corpus/train/linnaeus-IOBES/train.tsv',
                                                            './corpus/train/NCBI-IOBES/train.tsv'], help='path to training files')
    parser.add_argument('--dev_file', nargs='+', default=['./corpus/train/BC2GM-IOBES/devel.tsv',
                                                          './corpus/train/BC4CHEMD-IOBES/devel.tsv',
                                                          './corpus/train/BC5CDR-IOBES/devel.tsv',
                                                          './corpus/train/JNLPBA-IOBES/devel.tsv',
                                                          './corpus/train/linnaeus-IOBES/devel.tsv',
                                                          './corpus/train/NCBI-IOBES/devel.tsv'])
    parser.add_argument('--load_train_loader', help='path to pickle file for crf2train_dataloader')
    parser.add_argument('--load_c_train_loader', help='path to pickle file for combined crf2train_dataloader')
    parser.add_argument('--save_file', help='path to save data loader files')
    
    
    args = parser.parse_args()
    
    checkpoint_file = torch.load(args.checkpoint, map_location={'cuda:1':'cuda:'+str(torch.cuda.current_device())})
    train_args_all = json.load(open(args.train_args, 'r'))
    train_args = train_args_all['args']
    
    train_data_loader = pickle.load(open(args.load_train_loader, 'rb'))
    c_train_data_loader = pickle.load(open(args.load_c_train_loader, 'rb'))
    
    packer = CRFRepack_WC(len(train_args['tag2idx']), True)
    
    ner_model = LM_LSTM_CRF(len(train_args['tag2idx']), len(train_args['chr2idx']), 
        train_args['char_dim'], train_args['char_hidden'], train_args['char_layers'], 
        train_args['word_dim'], train_args['word_hidden'], train_args['word_layers'], 
        len(train_args['token2idx']), train_args['drop_out'], 1, 
        large_CRF=train_args['small_crf'], if_highway=train_args['high_way'], 
        in_doc_words=train_args['in_doc_words'], highway_layers = train_args['highway_layers'])
    
    ner_model.load_state_dict(checkpoint_file['state_dict'])
    #ner_model.cuda()
    
    decoder = CRFDecode_vb(len(train_args['tag2idx']), train_args['tag2idx']['<start>'], train_args['tag2idx']['<pad>'])
    
    
    new_crf2train_dataloader = make_silver(train_data_loader[0])
    
    model_name = args.checkpoint.split('/')[-1][:-6]
    if not os.path.exists(args.save_file + '/EXP1.1'):
        os.makedirs(args.save_file + '/EXP1.1')
    if not os.path.exists(args.save_file + '/EXP1.2'):
        os.makedirs(args.save_file + '/EXP1.2')
    if not os.path.exists(args.save_file + '/EXP1.3'):
        os.makedirs(args.save_file + '/EXP1.3')
    if not os.path.exists(args.save_file + '/EXP1.4'):
        os.makedirs(args.save_file + '/EXP1.4')
    
    protocol = 2
    # memory issue
    print('dumping!')
    with open(args.save_file + '/EXP1.1/new_crf2train_dataloader.p', 'wb') as f:
        for r in new_crf2train_dataloader[0]:
            pickle.dump(r, f, protocol)
    
    with open(args.save_file + '/EXP1.2/new_crf2train_dataloader.p', 'wb') as f:
        for r in new_crf2train_dataloader[1]:
            pickle.dump(r, f, protocol)
    
    with open(args.save_file + '/EXP1.3/new_crf2train_dataloader.p', 'wb') as f:
        for r in new_crf2train_dataloader[2]:
            pickle.dump(r, f, protocol)
    
    with open(args.save_file + '/EXP1.4/new_crf2train_dataloader.p', 'wb') as f:
        for r in new_crf2train_dataloader[3]:
            pickle.dump(r, f, protocol)
    
    new_crf2train_dataloader = None # free up memory
    
    
    c_new_crf2train_dataloader = make_silver(c_train_data_loader[0])
    
    print('dumping!')
    with open(args.save_file + '/EXP1.1/c_new_crf2train_dataloader.p', 'wb') as f:
        for r in c_new_crf2train_dataloader[0]:
            pickle.dump(r, f, protocol)
    
    with open(args.save_file + '/EXP1.2/c_new_crf2train_dataloader.p', 'wb') as f:
        for r in c_new_crf2train_dataloader[1]:
            pickle.dump(r, f, protocol)
    
    with open(args.save_file + '/EXP1.3/c_new_crf2train_dataloader.p', 'wb') as f:
        for r in c_new_crf2train_dataloader[2]:
            pickle.dump(r, f, protocol)
    
    with open(args.save_file + '/EXP1.4/c_new_crf2train_dataloader.p', 'wb') as f:
        for r in c_new_crf2train_dataloader[3]:
            pickle.dump(r, f, protocol)
    
    
    
    
    
    
    
    