

nes1 = set()
with open('train.tsv', 'r') as f:
    for line in f:
        try:
            nes1.add(line.strip().split('\t')[1].split('-')[1])
        except:
            pass

print(sorted(nes1))

nes2 = set()
with open('devel.tsv', 'r') as f:
    for line in f:
        try:
            nes2.add(line.strip().split('\t')[1].split('-')[1])
        except:
            pass

print(sorted(nes2))

nes3 = set()
with open('test.tsv', 'r') as f:
    for line in f:
        try:
            nes3.add(line.strip().split('\t')[1].split('-')[1])
        except:
            pass

print(sorted(nes3))


# intersect = nes1.intersection(nes2).intersection(nes3)
# print(sorted([r for r in nes1 if r not in intersect]))
# print(sorted([r for r in nes2 if r not in intersect]))
# print(sorted([r for r in nes3 if r not in intersect]))



