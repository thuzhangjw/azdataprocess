f = open('./mergedEMR.txt', 'r')
lines = f.readlines()
f.close()

titlelist = lines[0].strip().split('\t')
cols = []

titleset = set()
for idx, val in enumerate(titlelist):
    if val in titleset:
        continue
    titleset.add(val)
    cols.append(idx)

f = open('./finalEMR.txt', 'w')
newls = []
for l in lines:
    llist = l.strip().split('\t')
    newl = ''
    for i in cols:
        newl += (llist[i] + '\t')
    newl = newl[:-1]
    newls.append(newl + '\n')

n = len(newls)
newls[n-1] = newls[n-1][:-1]
f.writelines(newls)
f.close()

