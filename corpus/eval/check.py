nes = set()
with open('BioNLP11ID-IOBES/train.tsv', 'r') as f:
    for line in f:
        try:
            nes.add(line.strip().split('\t')[1].split('-')[1])
        except:
            pass

print(nes)



nes = set()
with open('BioNLP13CG-IOBES/train.tsv', 'r') as f:
    for line in f:
        try:
            nes.add(line.strip().split('\t')[1].split('-')[1])
        except:
            pass

print(nes)



nes = set()
with open('CRAFT-IOBES/train.tsv', 'r') as f:
    for line in f:
        try:
            nes.add(line.strip().split('\t')[1].split('-')[1])
        except:
            pass

print(nes)


nes = set()
with open('CELLFINDER-IOBES/test.tsv', 'r') as f:
    for line in f:
        try:
            nes.add(line.strip().split('\t')[1].split('-')[1])
        except:
            pass

print(nes)


nes = set()
with open('CHEMPROT-IOBES/test.tsv', 'r') as f:
    for line in f:
        try:
            nes.add(line.strip().split('\t')[1].split('-')[1])
        except:
            pass

print(nes)