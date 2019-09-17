import numpy as np
from sklearn.model_selection import KFold

if __name__ == "__main__":
    
    np.random.seed(0)
    
    data_dir = '/auto/nlg-05/huan183/NewBioNer/corpus/conll2003'
    save_dir = '/auto/nlg-05/huan183/NewBioNer/corpus/conll2003_converted'
    # save_dir = '/auto/nlg-05/huan183/NewBioNer/corpus/conll2003_converted2'
    # save_dir = '/auto/nlg-05/huan183/NewBioNer/corpus/conll2003_converted3'
    
    # train_file = data_dir + '/' + 'eng.train.bioes.conll'
    # dev_file = data_dir + '/' + 'eng.dev.bioes.conll'
    # test_file = data_dir + '/' + 'eng.test.bioes.conll'
    
    # train_data = [r.strip() for r in open(train_file, 'r').readlines()]
    # train_data_label = ['ORG', 'PER', 'MISC', 'LOC']
    # label2idx = {k:v for v,k in enumerate(train_data_label)}
    # train_sents = [[], [], [], []]
    # curr_sent, curr_label = "", np.random.choice(train_data_label)
    # for line in train_data:
        # if line == '':
            # if curr_sent:
                # train_sents[label2idx[curr_label]].append(curr_sent+'\n')
                # curr_sent = ""
            # curr_label = np.random.choice(train_data_label)
        # else:
            # idx, word, _, _, label = line.split(' ')
            # if label != 'O':
                # label_type = label.split('-')[1]
                # if label_type != curr_label:
                # # if label_type == curr_label:
                    # label = 'O'
            # curr_sent += word.lower() + '\t' + label + '\n'
    
    # curr_sent = ["", "", "", ""]
    # for line in train_data:
        # if line == '':
            # if curr_sent:
                # for i in range(4):
                    # train_sents[i].append(curr_sent[i]+'\n')
                # curr_sent = ["", "", "", ""]
            # curr_label = np.random.choice(train_data_label)
        # else:
            # idx, word, _, _, label = line.split(' ')
            # for i, curr_label in enumerate(train_data_label):
                # if label != 'O':
                    # label_type = label.split('-')[1]
                    # if label_type != curr_label:
                    # if label_type != curr_label:
                        # curr_sent[i] += word + '\t' + 'O' + '\n'
                    # else:
                        # curr_sent[i] += word + '\t' + label + '\n'
                # else:
                    # curr_sent[i] += word + '\t' + 'O' + '\n'
            
    
    # for i in range(4):
        # with open(save_dir + '/' + train_data_label[i] + '/train.tsv', 'w') as f:
            # for line in train_sents[i]:
                # f.write(line)
        
        # with open(dev_file, 'r') as fin, \
             # open(save_dir + '/' + train_data_label[i] + '/devel.tsv', 'w') as fout:
            # for line in fin:
                # if line == '\n':
                    # fout.write(line)
                # else:
                    # idx, word, _, _, label = line.strip().split(' ')
                    # if label != 'O':
                        # label_type = label.split('-')[1]
                        # if label_type != train_data_label[i]:
                        # # if label_type == train_data_label[i]:
                            # label = 'O'
                    # fout.write(word.lower() + '\t' + label + '\n')
    
    with open(dev_file, 'r') as fin, \
         open(save_dir + '/devel.tsv', 'w') as fout:
        for line in fin:
            if line == '\n':
                fout.write(line)
            else:
                idx, word, _, _, label = line.strip().split(' ')
                fout.write(word.lower() + '\t' + label + '\n')
    
    
    with open(save_dir + '/test.tsv', 'r') as fin, \
         open(save_dir + '/LOC/test.tsv', 'w') as fout_loc, \
         open(save_dir + '/PER/test.tsv', 'w') as fout_per, \
         open(save_dir + '/ORG/test.tsv', 'w') as fout_org, \
         open(save_dir + '/MISC/test.tsv', 'w') as fout_misc:
        for line in fin:
            if line == '\n':
                [f.write(line) for f in [fout_loc, fout_per, fout_org, fout_misc]]
            else:
                word, label = line.strip().split('\t')
                if label == 'O':
                    [f.write(line) for f in [fout_loc, fout_per, fout_org, fout_misc]]
                elif 'LOC' in label:
                    fout_loc.write(word.lower() + '\t' + label + '\n')
                    [f.write(word.lower() + '\t' + 'O' + '\n') for f in [fout_per, fout_org, fout_misc]]
                elif 'PER' in label:
                    fout_per.write(word.lower() + '\t' + label + '\n')
                    [f.write(word.lower() + '\t' + 'O' + '\n') for f in [fout_loc, fout_org, fout_misc]]
                elif 'ORG' in label:
                    fout_org.write(word.lower() + '\t' + label + '\n')
                    [f.write(word.lower() + '\t' + 'O' + '\n') for f in [fout_per, fout_loc, fout_misc]]
                elif 'MISC' in label:
                    fout_misc.write(word.lower() + '\t' + label + '\n')
                    [f.write(word.lower() + '\t' + 'O' + '\n') for f in [fout_per, fout_loc, fout_org]]
                else:
                    assert False
    