import pandas as pd
import re


classes = ['不稳定型心绞痛', '冠状动脉粥样硬化', '房颤', '急性心肌梗死', '非ST段抬高型心肌梗死']

def shuffle_and_batch(data, batch_size, epochs, shuffle=True):
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


def convert_word2_onehot(word_labels):
    labels =[]
    for dis in word_labels:
        label = [0] * len(classes)
        label[classes.index(dis)] = 1
        labels.append(label)

    return labels 


def load_text_and_labels(path):
    df = pd.read_csv(path, sep='\t')
    texts = df['disease_his']
    word_labels = df['GB/T-codename']
    
    labels = convert_word2_onehot(word_labels)

    x_texts = [ re.sub(r'。', '', text) for text in texts ]
    return x_texts, labels  


def batch_iter_text(x_data, y_data, batch_size, epochs, max_length, shuffle=True):
    x = map(lambda a: a.strip().split(' '), x_data)
    padded_x = map(lambda a: a+['unknown']*(max_length - len(a)), x)
    data = zip(padded_x, y_data)
    return shuffle_and_batch(data, batch_size, epochs, shuffle)


def wordvec2str(word, vec):
    res = word
    for i in list(vec):
        res += ' ' + str(i)
    return res


def load_numeric_feature_and_labels(path):
    df = pd.read_csv(path, sep='\t')
    numeric_features = df.iloc[:, 3:].astype('float32')
    word_labels = df['GB/T-codename']
    labels = convert_word2_onehot(word_labels)

    return numeric_features.values.tolist(), labels 


def batch_iter_numeric(x_data, y_data, batch_size, epochs, shuffle=True):
    data = zip(x_data, y_data)
    return shuffle_and_batch(data, batch_size, epochs, shuffle)


def load_all_data(path):
    df = pd.read_csv(path, sep='\t')
    texts = df['disease_his']
    texts_feature = [re.sub(r'。', '', text) for text in texts]

    numeric_features = df.iloc[:, 3:].astype('float32').values.tolist()

    word_labels = df['GB/T-codename']
    labels = convert_word2_onehot(word_labels)
    return texts_feature, numeric_features, labels 

# need unzip two times
def batch_iter(text_features, numeric_features, labels, batch_size, epochs, max_length, shuffle=True):
    x_data = text_features
    y_data = zip(numeric_features, labels)
    return batch_iter_text(x_data, y_data, batch_size, epochs, max_length, shuffle) 

