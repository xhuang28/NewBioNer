import torch, argparse, json, pickle, itertools
from model.lm_lstm_crf import *
from model.data_util import *
from model.crf import *
from pathlib import Path


def load_data(train_features, train_labels, train_args):
    crf2train_dataloader = build_crf2dataloader(train_args['crf2corpus'], train_features, train_labels, train_args['batch_size'], train_args['corpus_missing_tagspace'], train_args['corpus_mask_value'], train_args['tag2idx'], train_args['chr2idx'], train_args['token2idx'], train_args['caseless'], shuffle=False, drop_last=False) 
    return crf2train_dataloader



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
    parser.add_argument('--load_pickle', default='crf2train_dataloader.pickle', help='path to pickle file for crf2train_dataloader')
    
    
    args = parser.parse_args()
    
    checkpoint_file = torch.load(args.checkpoint, map_location={'cuda:1':'cuda:'+str(torch.cuda.current_device())})
    train_args = json.load(open(args.train_args, 'r'))['args']
    
    train_features, train_labels = read_combine_data(args.train_file, args.dev_file)
    
    ##### <copied from original code> #####
    corpus_missing_tagspace = build_corpus_missing_tagspace(train_labels, train_args['tag2idx'])
    corpus2crf, corpus_str2crf = corpus_dispatcher(corpus_missing_tagspace, style='N21')
    crf2corpus = {}
    for key, val in corpus2crf.items():
        if val not in crf2corpus:
            crf2corpus[val] = [key]
        else:
            crf2corpus[val] += [key]
    
    train_args['crf2corpus'] = crf2corpus
    ##### </copied from original code> #####
    
    
    if args.load_pickle == False or not Path(args.load_pickle).is_file():
        crf2train_dataloader = load_data(args.train_file, args.dev_file, train_args, True)
        pickle.dump(crf2train_dataloader, open('crf2train_dataloader.pickle', 'wb'))
    else:
        crf2train_dataloader = pickle.load(open(args.load_pickle, 'rb'))
    
    
    packer = CRFRepack_WC(len(train_args['tag2idx']), True)
    
    ner_model = LM_LSTM_CRF(len(train_args['tag2idx']), len(train_args['chr2idx']), 
        train_args['char_dim'], train_args['char_hidden'], train_args['char_layers'], 
        train_args['word_dim'], train_args['word_hidden'], train_args['word_layers'], 
        len(train_args['token2idx']), train_args['drop_out'], len(train_args['crf2corpus']), 
        large_CRF=train_args['small_crf'], if_highway=train_args['high_way'], 
        in_doc_words=train_args['in_doc_words'], highway_layers = train_args['highway_layers'])
    
    ner_model.load_state_dict(checkpoint_file['state_dict'])
    #ner_model.cuda()
    
    for f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v, corpus_mask_v, reorder in itertools.chain.from_iterable(crf2train_dataloader[0]):
        f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, corpus_mask_v = packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v, corpus_mask_v)
        scores = ner_model(f_f.cpu(), f_p.cpu(), b_f.cpu(), b_p.cpu(), w_f.cpu(), 0, corpus_mask_v.cpu())
        pickle.dump(scores, open('save_point_1.pickle', 'wb'), 1)
        break
    
    
    
    
    
    
    
    
    