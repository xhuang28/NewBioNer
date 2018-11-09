import argparse
import os
import sys
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning with LM-LSTM-CRF together with Language Model')
    # files
    parser.add_argument('--train_file', nargs='+', default='./corpus/BC5CDR-IOBES/train.tsv', help='path to training file')
    parser.add_argument('--test_as_train', nargs='+', help='path to test file as train file')
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
    parser.add_argument('--idx_combination', type=int, default=0)
    
    
    args = parser.parse_args()
    
    infix = str(args.idx_combination)
    pickle.dump(args, open(args.data_loader + '/P3' + infix + '_args' + '.p', 'wb'), 0)