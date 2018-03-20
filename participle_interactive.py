import pandas as pd
import jieba
import sys
import os 

df = pd.read_csv('../data/undersampled.txt', sep='\t')
jieba.load_userdict('./mydict.txt')
jieba.load_userdict('./my_dict_from_other_source.txt')
def load_suggest_freq():
    if os.path.exists('./suggest_freq.txt'):
        f = open('./suggest_freq.txt', 'r')
        lines = f.readlines()
        for line in lines:
             words = line.split(' ')
             jieba.suggest_freq((words[0], words[1]), True)
        f.close()

load_suggest_freq()


dict_f = open('./mydict.txt', 'a')

words_list = []
line = 0
if os.path.exists('./participle_log'):
    with open('./participle_log', 'r') as f:
        line = int(f.readline())

if line >= len(df):
    print('participle has completed')
    exit(0)

end = False
for idx, text in enumerate(df['现病史']):
    if idx < line:
        continue
    print(df.loc[idx, 'GB/T-codename'])
    print(text, '\n')
    flag = 'n'
    while flag != 'y':
        seg_list = jieba.lcut(text)
        print(seg_list)
        s = input()
        if s == '' or s == 'end':
            flag = 'y'
            if s == 'end':
                end = True
        else:
            dict_list = s.split('/')
            for d in dict_list:
                r = d.split(' ')
                jieba.add_word(r[0], freq=int(r[1]))
                dict_f.write(r[0] + ' ' + r[1] + '\n')
    
    words_list += seg_list
    if end:
        line = idx + 1
        break

dict_f.close()
from collections import Counter

c = Counter(words_list)
sorted_words = sorted(c.items(), key=lambda x: x[1], reverse=True)

with open(sys.argv[1], 'w') as f:
    for w in sorted_words:
        f.write(w[0] + ' ' + str(w[1]) + '\n')


with open('./participle_log', 'w') as f:
    f .write(str(line))

