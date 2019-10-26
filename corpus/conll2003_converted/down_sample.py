import numpy as np

np.random.seed(999)

data_path = [['', '/auto/nlg-05/huan183/NewBioNer/corpus/conll2003_converted'], \
             ['LOC', '/auto/nlg-05/huan183/NewBioNer/corpus/conll2003_converted/LOC'], \
             ['MISC', '/auto/nlg-05/huan183/NewBioNer/corpus/conll2003_converted/MISC'], \
             ['ORG', '/auto/nlg-05/huan183/NewBioNer/corpus/conll2003_converted/ORG'], \
             ['PER', '/auto/nlg-05/huan183/NewBioNer/corpus/conll2003_converted/PER']]

for data_name, path in data_path:
    lines = open(path+'/train.tsv', 'r').readlines()
    sents, curr_sent = [], []
    for line in lines:
        curr_sent.append(line)
        if line == '\n':
            if len(curr_sent) > 1:
                sents.append(curr_sent)
            curr_sent = []
    
    for n_samples in [50, 100, 150, 200, 250, 300, 350, 400, 600, 800, 1000, 2000]:
        assert len(sents) > n_samples
        down_sampled = np.random.choice(sents, n_samples, replace=False)
        with open(path+'/train_'+str(n_samples)+'.tsv', 'w') as f:
            for sent in down_sampled:
                f.write(''.join(sent))