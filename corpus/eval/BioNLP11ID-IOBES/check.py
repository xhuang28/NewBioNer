

nes = set()
with open('train.tsv', 'r') as f:
    for line in f:
        try:
            nes.add(line.strip().split('\t')[1].split('-')[1])
        except:
            pass

print(nes)

nes = set()
with open('devel.tsv', 'r') as f:
    for line in f:
        try:
            nes.add(line.strip().split('\t')[1].split('-')[1])
        except:
            pass

print(nes)

nes = set()
with open('test.tsv', 'r') as f:
    for line in f:
        try:
            nes.add(line.strip().split('\t')[1].split('-')[1])
        except:
            pass

print(nes)
