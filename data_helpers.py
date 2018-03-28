import pandas as pd
import re


classes = ['不稳定型心绞痛', '冠状动脉粥样硬化', '房颤', '急性心肌梗死', '非ST段抬高型心肌梗死']


def load_text_and_labels(path):
    df = pd.read_csv(path, sep='\t')
    texts = df['disease_his']
    word_labels = df['GB/T-codename']
    
    labels = []
    for dis in word_labels:
        label = [0] * len(classes)
        label[classes.index(dis)] = 1
        labels.append(label)

    x_texts = [ re.sub(r'。', '', text) for text in texts ]
    return x_texts, labels  


def batch_iter(x_data, y_data, batch_size, epochs, max_length, shuffle=True):
    x = map(lambda a: a.strip().split(' '), x_data)
    padded_x = map(lambda a: a+['unknown']*(max_length - len(a)), x)
    data = zip(padded_x, y_data)
    s = pd.Series(list(data))
    data_size = len(s)
    batch_nums = int((len(s) - 1) / batch_size) + 1
    res = []
    for _ in range(epochs):
        if shuffle:
            shuffled_data = s.sample(frac=1)
        else:
            shuffled_data = s
        for batch_num in range(batch_nums):
            start_index = batch_num * batch_size
            end_index = min(data_size, (batch_num + 1)* batch_size)
            res.append(list(shuffled_data[start_index:end_index]))
    return res 


def wordvec2str(word, vec):
    res = word
    for i in list(vec):
        res += ' ' + str(i)
    return res

