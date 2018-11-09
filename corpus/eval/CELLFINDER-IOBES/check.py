

nes3 = set()
with open('test.tsv', 'r') as f:
    for line in f:
        try:
            nes3.add(line.strip().split('\t')[1].split('-')[1])
        except:
            pass

print(sorted(nes3))




