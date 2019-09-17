import torch, json
from model.lm_lstm_crf import *
from model.data_util import *
from model.utils import *
from model.crf import *
from model.predictor import *
from model.evaluator import *
import model.utils_old as utils_old
import model.evaluator_old as Evaluator


checkpoint = '/auto/nlg-05/huan183/NewBioNer/checkpoints/P1/EXP2/CONLL_MTL_C100_W300/N2N_LOC_LAST_0.9900_0.9865_0.9935_300'

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

args = json.load(open(checkpoint+'.json', 'r'))['args']
args = Struct(**args)
args.idx2tag = {k:v for v,k in args.tag2idx.items()}
ner_model = LM_LSTM_CRF(len(args.tag2idx), len(args.chr2idx), 
    args.char_dim, args.char_hidden, args.char_layers, 
    args.word_dim, args.word_hidden, args.word_layers, len(args.token2idx), 
    args.drop_out, len(args.corpus2crf), 
    large_CRF=args.small_crf, if_highway=args.high_way, 
    in_doc_words=args.in_doc_words, highway_layers = args.highway_layers, sigmoid = args.sigmoid)
ner_model.load_state_dict(torch.load(checkpoint+'.model')['state_dict'])


data_dir = '/auto/nlg-05/huan183/NewBioNer/corpus/conll2003_converted/'

for crf_no, test_file_dir in [[0,'LOC/'], [1,'MISC/'], [2,'ORG/'], [3,'PER/']]:
    test_features, test_labels = read_data([data_dir+test_file_dir+'test.tsv'])
    test_dataset_loader = []
    for i in range(1):
        test_missing_tagspace = build_corpus_missing_tagspace(test_labels, args.tag2idx)
        test_missing_tagspace = test_missing_tagspace * len(test_labels[i])
        test_dataset_loader.append(build_dataloader(test_features[i], test_labels[i], 50, test_missing_tagspace, args.corpus_mask_value, args.tag2idx, args.chr2idx, args.token2idx, args.caseless, shuffle=False, drop_last=False))


    evaluator = Evaluator.eval_sentence()
    packer = CRFRepack_WC(len(args.tag2idx), True)
    predictor = Predictor(args.tag2idx, packer, label_seq = True, batch_size = 50)
    ner_model.cuda()
    ner_model.eval()

    preds, labels = [], []
    for f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v, corpus_mask_v, reorder in itertools.chain.from_iterable(test_dataset_loader[0]):
        f_f, f_p, b_f, b_p, w_f, _, mask_v, corpus_mask_v = predictor.packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v, corpus_mask_v, volatile=True)
        pred, scores = predictor.predict_batch(ner_model, crf_no, f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v, corpus_mask_v, "M")
        preds += torch.unbind(pred, 1)
        labels += torch.unbind((tg / len(args.tag2idx)).view(tg.shape[0], tg.shape[1]), 0)

    for i in range(len(preds)):
        curr_preds = [args.idx2tag[r] for r in preds[i].numpy().tolist() if r!=args.tag2idx['<pad>']]
        curr_labels = [args.idx2tag[r] for r in labels[i].numpy().tolist() if not r in [args.tag2idx['<pad>'], args.tag2idx['<start>']]]
        evaluator.eval_sent(curr_preds, curr_labels)

    f1, prec, recall, acc = evaluator.f1_score()
    print("Results on %s:" % (data_dir+test_file_dir+'test.tsv'))
    print("\nf1:\t%.4f\nprec:\t%.4f\nrecall:\t%.4f\nacc:\t%.4f\n" % (f1, prec, recall, acc))