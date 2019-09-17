import argparse, pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='path to data directory')
    
    args = parser.parse_args()
    
    train_dirs = [args.data_dir + '/train/' + r for r in [rr+'-IOBES' for rr in ['BC2GM', 'BC4CHEMD', 'BC5CDR', 'JNLPBA', 'linnaeus', 'NCBI']]]
    test_dirs = [args.data_dir + '/eval/' + r for r in [rr+'-IOBES' for rr in ['BioNLP11ID', 'BioNLP13CG', 'CELLFINDER', 'CHEMPROT', 'CRAFT']]]
    
    max_sent_len = 0
    for d in train_dirs:
        for f in ['train', 'devel', 'test']:
            lines = open(d+'/'+f+'.tsv', 'r').readlines()
            lines = [r.strip() if r.strip()!='' else '\n' for r in lines]
            lines = [r.split('\t') if r!='\n' else ['\n']*2 for r in lines]
            words = [r[0] for r in lines]
            tags = [r[1] for r in lines]
            sents = ' '.join(words).split(' \n ')
            sent_tags = [r.split(' ') for r in ' '.join(tags).split(' \n ')]
            curr_max_sent_len = max([len(r) for r in sent_tags])
            if curr_max_sent_len > max_sent_len:
                max_sent_len = curr_max_sent_len
            pickle.dump(sent_tags, open(d+'/'+f+'_tags.p', 'wb'))
            with open(d+'/'+f+'_sents.txt', 'w') as fout:
                for s in sents:
                    fout.write(s+'\n')
    
    print('max sent length:', max_sent_len)