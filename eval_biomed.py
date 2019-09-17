import torch, json
from model.lm_lstm_crf import *
from model.data_util import *
from model.utils import *
from model.crf import *
from model.predictor import *
from model.evaluator import *
import model.utils_old as utils_old
import model.evaluator_old as Evaluator


def combine_seq(seq1, seq2):
    # seq1 has higher priority, and we retain all chunks in seq1.
    # We add chunks in seq2 if it does not conflict with chunks in seq1.
    seq1_chunks = utils_old.iobes_to_spans(seq1)
    seq1_chunks = [set(chunk.split('@')[1:]) for chunk in seq1_chunks]
    seq2_chunks = utils_old.iobes_to_spans(seq2)
    seq2_chunks = [set(chunk.split('@')[1:]) for chunk in seq2_chunks]
    output_seq = seq1[:]
    seq1_chunk_all = set()
    for s in seq1_chunks:
        seq1_chunk_all = seq1_chunk_all.union(s)
    for spans in seq2_chunks:
        if not spans.intersection(seq1_chunk_all):
            for idx in spans:
                output_seq[int(idx)] = seq2[int(idx)]
    return output_seq

def combine_prediction(seperate_predictions):
    """
    Combine separated preditions on different CDRS
    :param seperate_predictions: (path_score, decode_path)
    :return: a combined prediction
    """
    seperate_predictions.sort()
    combined = seperate_predictions[0][1][:]  # first prediction sequence
    for idx in range(1, len(seperate_predictions)):
        label_seq = seperate_predictions[idx][1]
        combined = combine_seq(label_seq, combined)
    return combined


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


# checkpoint = '/auto/nlg-05/huan183/NewBioNer/checkpoints/P1/EXP2/100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63'
# checkpoint = '/auto/nlg-05/huan183/NewBioNer/checkpoints/P1/EXP2/100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42'
checkpoint = '/auto/nlg-05/huan183/NewBioNer/checkpoints/P1/EXP2/100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86'

args = json.load(open(checkpoint+'.json', 'r'))['args']
args = Struct(**args)
args.idx2tag = {k:v for v,k in args.tag2idx.items()}
ner_model = LM_LSTM_CRF(len(args.tag2idx), len(args.chr2idx), 
    args.char_dim, args.char_hidden, args.char_layers, 
    args.word_dim, args.word_hidden, args.word_layers, len(args.token2idx), 
    args.drop_out, 1, 
    large_CRF=args.small_crf, if_highway=args.high_way, 
    in_doc_words=args.in_doc_words, highway_layers = args.highway_layers, sigmoid = args.sigmoid)
ner_model.load_state_dict(torch.load(checkpoint+'.model')['state_dict'])

evaluator = Evaluator.eval_sentence()
packer = CRFRepack_WC(len(args.tag2idx), True)
predictor = Predictor(args.tag2idx, packer, label_seq = True, batch_size = 50)
ner_model.cuda()
ner_model.eval()



# data_files = [['BioNLP11ID', '/auto/nlg-05/huan183/NewBioNer/corpus/eval/BioNLP11ID-IOBES/test.tsv'], \
              # ['BioNLP13CG', '/auto/nlg-05/huan183/NewBioNer/corpus/eval/BioNLP13CG-IOBES/test.tsv'], \
              # ['CELLFINDER', '/auto/nlg-05/huan183/NewBioNer/corpus/eval/CELLFINDER-IOBES/test.tsv'], \
              # ['CHEMPROT', '/auto/nlg-05/huan183/NewBioNer/corpus/eval/CHEMPROT-IOBES/test.tsv'], \
              # ['CRAFT', '/auto/nlg-05/huan183/NewBioNer/corpus/eval/CRAFT-IOBES/test.tsv'], \
              # ['BC5CDR', '/auto/nlg-05/huan183/NewBioNer/corpus/eval/BC5CDR-IOBES/test.tsv']]

data_files = [['BC2GM', '/auto/nlg-05/huan183/NewBioNer/corpus/train/BC2GM-IOBES/test.tsv'], \
              ['BC4CHEMD', '/auto/nlg-05/huan183/NewBioNer/corpus/train/BC4CHEMD-IOBES/test.tsv'], \
              ['JNLPBA', '/auto/nlg-05/huan183/NewBioNer/corpus/train/JNLPBA-IOBES/test.tsv'], \
              ['linnaeus', '/auto/nlg-05/huan183/NewBioNer/corpus/train/linnaeus-IOBES/test.tsv'], \
              ['NCBI', '/auto/nlg-05/huan183/NewBioNer/corpus/train/NCBI-IOBES/test.tsv']]



for data_name, data_file in data_files:

    test_features, test_labels = read_data([data_file])
    test_labels = [[[rr if rr in args.tag2idx else 'O' for rr in r] for r in test_labels[0]]]
    test_dataset_loader = []
    for i in range(1):
        test_missing_tagspace = build_corpus_missing_tagspace(test_labels, args.tag2idx)
        test_missing_tagspace = test_missing_tagspace * len(test_labels[i])
        test_dataset_loader.append(build_dataloader(test_features[i], test_labels[i], 50, test_missing_tagspace, args.corpus_mask_value, args.tag2idx, args.chr2idx, args.token2idx, args.caseless, shuffle=False, drop_last=False))

    preds, labels = [], []
    for crf_no in range(1):
        curr_preds, curr_labels = [], []
        for f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v, corpus_mask_v, reorder in itertools.chain.from_iterable(test_dataset_loader[0]):
            f_f, f_p, b_f, b_p, w_f, _, mask_v, corpus_mask_v = predictor.packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v, corpus_mask_v, volatile=True)
            pred, scores = predictor.predict_batch(ner_model, crf_no, f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v, corpus_mask_v, "M")
            curr_preds += torch.unbind(pred, 1)
            curr_labels += torch.unbind((tg / len(args.tag2idx)).view(tg.shape[0], tg.shape[1]), 0)
        preds.append(curr_preds)
        labels.append(curr_labels)

    for i in range(len(preds[0])):
        curr_preds = [(0, [args.idx2tag[rr] for rr in r[i].numpy().tolist() if rr!=args.tag2idx['<pad>']]) for r in preds]
        curr_labels = [args.idx2tag[rr] for rr in labels[0][i].numpy().tolist() if not rr in [args.tag2idx['<pad>'], args.tag2idx['<start>']]]
        combined_pred = combine_prediction(curr_preds)
        evaluator.eval_sent(combined_pred, curr_labels)

    f1, prec, recall, acc = evaluator.f1_score()
    print("Dataset:", data_name)
    print("\nf1:\t%.4f\nprec:\t%.4f\nrecall:\t%.4f\nacc:\t%.4f\n" % (f1, prec, recall, acc))