import sys
import re
import pandas as pd

lmap = {}

f = open(sys.argv[1], 'r')
lines = f.readlines()
f.close()

for line in lines:
    comma_sentences = re.split('[,，]', line)
    for sentence in comma_sentences:
        word_count = len(sentence.strip().split(' '))
        if word_count in lmap:
            lmap[word_count][0][0] += 1
            lmap[word_count][1].append(sentence)
        else:
            lmap[word_count] = ([1], [sentence])

reslist = sorted(lmap.items(), key=lambda x: x[0])
for res in reslist:
    print(res[0], res[1][0][0])

s = input()
while s != 'end':
    print(lmap[int(s)][1])
    s = input()

