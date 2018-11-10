nes = set()
with open('BC2GM-IOBES/train.tsv', 'r') as f:
    for line in f:
        try:
            nes.add(line.strip().split('\t')[1].split('-')[1])
        except:
            pass

print(nes)



nes = set()
with open('BC4CHEMD-IOBES/train.tsv', 'r') as f:
    for line in f:
        try:
            nes.add(line.strip().split('\t')[1].split('-')[1])
        except:
            pass

print(nes)



nes = set()
with open('BC5CDR-IOBES/train.tsv', 'r') as f:
    for line in f:
        try:
            nes.add(line.strip().split('\t')[1].split('-')[1])
        except:
            pass

print(nes)


nes = set()
with open('JNLPBA-IOBES/test.tsv', 'r') as f:
    for line in f:
        try:
            nes.add(line.strip().split('\t')[1].split('-')[1])
        except:
            pass

print(nes)


nes = set()
with open('linnaeus-IOBES/test.tsv', 'r') as f:
    for line in f:
        try:
            nes.add(line.strip().split('\t')[1].split('-')[1])
        except:
            pass

print(nes)


nes = set()
with open('NCBI-IOBES/test.tsv', 'r') as f:
    for line in f:
        try:
            nes.add(line.strip().split('\t')[1].split('-')[1])
        except:
            pass

print(nes)