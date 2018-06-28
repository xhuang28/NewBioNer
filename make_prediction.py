import torch, argparse, json, pickle
from model.lm_lstm_crf import *
from model.data_util import *


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
    parser.add_argument('--dev_file', nargs='+', default=['./corpus/train/BC2GM-IOBES/train.tsv',
                                                          './corpus/train/BC4CHEMD-IOBES/train.tsv',
                                                          './corpus/train/BC5CDR-IOBES/train.tsv',
                                                          './corpus/train/JNLPBA-IOBES/train.tsv',
                                                          './corpus/train/linnaeus-IOBES/train.tsv',
                                                          './corpus/train/NCBI-IOBES/train.tsv'])
    
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
    
    print(train_args['corpus2crf'])
    print(crf2corpus)
    ##### </copied from original code> #####
    
    
    crf2train_dataloader = build_crf2dataloader(crf2corpus, train_features, train_labels, train_args['batch_size'], train_args['corpus_missing_tagspace'], train_args['corpus_mask_value'], train_args['tag2idx'], train_args['chr2idx'], train_args['token2idx'], train_args['caseless'], shuffle=True, drop_last=False) 
    
    pickle.dump(crf2train_dataloader, open('crf2train_dataloader.pickle', 'wb'))
    print(len(self.crf2corpus) - 1)
    print(len(crf2train_dataloader))
    crf2train_dataloader = pickle.load(open('crf2train_dataloader.pickle', 'rb'))
    print('load successful')
    
    # for f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v, corpus_mask_v, reorder in crf2train_dataloader[0]
    
    # ner_model = LM_LSTM_CRF(len(train_args['tag2idx']), len(train_args['chr2idx']), 
        # train_args['char_dim'], train_args['char_hidden'], train_args['char_layers'], 
        # train_args['word_dim'], train_args['word_hidden'], train_args['word_layers'], 
        # len(train_args['token2idx']), train_args['drop_out'], len(train_args['crf2corpus']), 
        # large_CRF=train_args['small_crf'], if_highway=train_args['high_way'], 
        # in_doc_words=train_args['in_doc_words'], highway_layers = train_args['highway_layers'])
    
    # train_features, train_labels = read_data(args.train_file)
    
    
    
    
    