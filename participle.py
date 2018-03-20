import pandas as pd
import jieba
import sys
import os 


df = pd.read_csv(sys.argv[1], sep ='\t')
jieba.load_userdict('./my_dict_from_other_source.txt')
jieba.load_userdict('./mydict.txt')

stopwords = set()

with open('./stopword-full.dic', 'r') as f:
    lines = f.readlines()
    for word in lines:
        stopwords.add(word.strip())


def load_suggest_freq():
    if os.path.exists('./suggest_freq.txt'):
        f = open('./suggest_freq.txt', 'r')
        lines = f.readlines()
        for line in lines:
             words = line.split(' ')
             jieba.suggest_freq((words[0], words[1]), True)
        f.close()

load_suggest_freq()
sentence_list = []
for text in df['现病史']:
    print(text, '\n')
    seg_list = jieba.lcut(text)
    s = ''
    for word in seg_list:
        if word not in stopwords:
            s += word + ' ';
    s = s.strip()
    print(s, '\n')
    sentence_list.append(s)
    input()

#from collections import Counter
#
#c = Counter(words_list)
#sorted_words = sorted(c.items(), key=lambda x: x[1], reverse=True)
#
#with open(sys.argv[2], 'w') as f:
#    for w in sorted_words:
#        f.write(w[0]+ ' ' + str(w[1]) + '\n')

newdf = pd.DataFrame({'disease_his' : sentence_list})
newdf = pd.concat([newdf, df], axis=1)

newdf.to_csv('../data/participled.txt', sep='\t', index=False)

