import pandas as pd
import jieba
import sys
import os 
import re 

df = pd.read_csv(sys.argv[1], sep ='\t')
jieba.load_userdict('./my_dict_from_other_source.txt')
jieba.load_userdict('./mydict.txt')
rs = [re.compile('[(（][^(（]*[)）]'), re.compile('患者自发病以来.*。')]
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

neg_words=['无']
def add_negtive_word(wordslist):
    preprocessd = list(map(lambda x: [x, '、'][x == '及'], wordslist))
#    preprocessd = wordslist 
    pos = []
    for idx, word in enumerate(preprocessd):
        for nw in neg_words:
            if word == nw:
                for i in range(idx+1, len(preprocessd)-1):
                    if preprocessd[i] == '、' and preprocessd[i+1] not in neg_words:
                        pos.append((i+1, nw))
                    elif preprocessd[i] == ',' or preprocessd[i] == '，' or preprocessd[i] == '.' or preprocessd[i] == '。':
                        break
    res = preprocessd
    for idx, p in enumerate(pos):
        res.insert(idx + p[0], p[1])
    return res 


load_suggest_freq()
sentence_list = []
word_list_list = []
for text in df['现病史']:
#    print(text, '\n')
    for r in rs:
        text = r.sub('', text)
    sentences = text.split('。')
    newdoc = ''
    for s in sentences:
        news = ''
        if s == '':
            continue
        seg_list = jieba.lcut(s)
        processd_list = add_negtive_word(seg_list)
#        processd_list = seg_list 
        for word in processd_list:
            if word not in stopwords:
                news += word + ' '
        news += '。 '
        word_list_list.append(news.strip('。 '))
        newdoc += news 
    newdoc = newdoc.strip()
#    print(newdoc, '\n')
    sentence_list.append(newdoc)
#    input()

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

with open(sys.argv[2], 'w') as f:
    for s in word_list_list:
        f.write(s + '\n')

