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


def load_sentences_matrix__and_labels(path):
    df = pd.read_csv(path, sep='\t')
    texts = df['disease_his']
    word_labels = df['GB/T-codename']
    labels = convert_word2_onehot(word_labels)
    x_texts = [text.split('。') for text in texts]
    x_texts = [ [text, text[:-1]][text[-1]=='']  for text in x_texts ]
    return x_texts, labels


def batch_iter_text_matrix(x_data, y_data, batch_size, epochs, max_sentence_length, max_sentence_num, shuffle=True):
    sentence_num_list = [len(a) for a in x_data]
    x = []
    for doc in x_data:
        tmp = []
        for sentence in doc:
            ns = sentence.strip().split(' ')
            ns += ['unknown'] * (max_sentence_length - len(ns))
            tmp.append(ns)
        for _ in range(max_sentence_num - len(doc)):
            tmp.append(['unknown']*max_sentence_length)
        x.append(tmp)
    z = list(zip(x, sentence_num_list))
    data = zip(z, y_data)
    return shuffle_and_batch(data, batch_size, epochs, shuffle)


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


def batch_iter_whole(x_cnn, x_ncnn, x_cnn_rnn, y, batch_size, epochs, cnn_max_sentence_length, cnn_rnn_max_sentence_num, cnn_rnn_max_sentence_length, shuffle=True):
    x = map(lambda a: a.strip().split(' '), x_cnn)
    padded_x_cnn = map(lambda a: a+['unknown']*(cnn_max_sentence_length - len(a)), x)
    
    sentence_num_list = [len(a) for a in x_cnn_rnn]
    padded_x_cnn_rnn = []
    for doc in x_cnn_rnn:
        tmp = []
        for sentence in doc:
            ns = sentence.strip().split(' ')
            ns += ['unknown'] * (cnn_rnn_max_sentence_length - len(ns))
            tmp.append(ns)
        for _ in range(cnn_rnn_max_sentence_num - len(doc)):
            tmp.append(['unknown']*cnn_rnn_max_sentence_length)
        padded_x_cnn_rnn.append(tmp) 

    data = zip(padded_x_cnn, padded_x_cnn_rnn, x_ncnn, sentence_num_list, y)
    return shuffle_and_batch(data, batch_size, epochs, shuffle)


def write_word2vec_text(path, vocab, keys):
    with open(path, 'w') as f:
        f.write(str(len(vocab)) + ' ' + str(len(vocab[0])))
        idx = 0
        for val in vocab:
            if idx == 0:
                f.write('\n' + wordvec2str('unknown', val))
            else:
                f.write('\n' + wordvec2str(keys[idx-1], val))
            idx += 1


def reshape2matrix(line_feature, feature_dim_list, max_feature_dim):
    i = 0
    res = []
    for l in feature_dim_list:
        tmp = [0.0] * i
        tmp += line_feature[i:i+l]
        tmp += [0.0] * (max_feature_dim - (i+l))
        res.append(tmp)
        i += l
    assert(i == len(line_feature))
    return res
#    for l in feature_dim_list:
#        tmp = line_feature[i:i+l]
#        tmp += [0.0] * (max_feature_dim - l)
#        res.append(tmp)
#        i += l
#    assert(i == len(line_feature))
#    return res



def generate_feature_dim_list(column_names):
    feature_dim_list = []
    current_name = column_names[0].split('_')[0]
    l = 0
    for name in column_names:
        tn = name.split('_')[0]
        if tn == current_name:
            l += 1
        else:
            feature_dim_list.append(l)
            l = 1
            current_name = tn
    feature_dim_list.append(l)
    assert(len(column_names) == sum(feature_dim_list))
    return feature_dim_list


def load_numeric_matrix_data(path):
    df = pd.read_csv(path, sep='\t')
    feature_dim_list = generate_feature_dim_list(df.columns[3:])
    max_feature_dim = sum(feature_dim_list)
    numeric_features = df.iloc[:, 3:].astype('float32').values.tolist()
    word_labels = df['GB/T-codename']
    labels = convert_word2_onehot(word_labels)
    
    assert(len(numeric_features[0]) == max_feature_dim)
    numeric_matrix_features = list(map(lambda x: reshape2matrix(x, feature_dim_list, max_feature_dim), numeric_features))
    return numeric_matrix_features, labels, feature_dim_list
#    df = pd.read_csv(path, sep='\t')
#    feature_dim_list = generate_feature_dim_list(df.columns[3:])
#    max_feature_dim = max(feature_dim_list)
#    numeric_features = df.iloc[:, 3:].astype('float32').values.tolist()
#    word_labels = df['GB/T-codename']
#    labels = convert_word2_onehot(word_labels)
#    
#    assert(len(numeric_features[0]) == sum(feature_dim_list))
#    numeric_matrix_features = list(map(lambda x: reshape2matrix(x, feature_dim_list, max_feature_dim), numeric_features))
#    return numeric_matrix_features, labels
    

def get_cnn_rnn_max_sentence_length(x_train, x_test):
    res = 0
    for doc in (x_train + x_test):
        for sentence in doc:
            sl = len(sentence.strip().split(' '))
            if sl > res:
                res = sl
    return res


def save_confusion_matrix(reals, predictions, path):
    num_classes = len(classes)
    d = {}
    for dis in classes:
        d[dis] = pd.Series([0]*num_classes, index=classes)
    
    confusion_matrix = pd.DataFrame(d)
    for i in range(len(reals)):
        ri = int(reals[i])
        pi = int(predictions[i])
        confusion_matrix.iloc[ri, pi] += 1
    confusion_matrix.to_csv(path, sep='\t')

