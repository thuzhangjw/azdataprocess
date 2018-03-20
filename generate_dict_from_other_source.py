import os

mydict = set()

dirpath = '../dict'
n = 0
for filename in os.listdir(dirpath):
    filepath = os.path.join(dirpath, filename)
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for w in lines:
            mydict.add(w.strip().split(' ')[0])
            n += 1

print('total: ', n)

with open('my_dict_from_other_source.txt', 'w') as f:
    for w in mydict:
        if w != '':
            f.write(w + ' 10000\n')


