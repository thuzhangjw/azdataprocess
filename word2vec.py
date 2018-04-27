import gensim
import sys
import copy

sentences = gensim.models.word2vec.LineSentence('../data2/corups.txt')
#model = gensim.models.word2vec.Word2Vec(min_count=1, window=5, size=100, workers=16)
#model.build_vocab(sentences)
#
#loss = sys.maxsize
#pos = 0
#
#for i in range(10):
#    model.train(sentences, total_examples=model.corpus_count, epochs=1, compute_loss=True)
#    nloss = model.get_latest_training_loss()
#    if nloss < loss:
#        loss = nloss
#        final_model = copy.deepcopy(model)
#        pos = i + 1
#
#
#print('loss:', loss, ' at iter: ', pos)
#
#final_model.save('../data/word2vec.model')
#

model = gensim.models.word2vec.Word2Vec(sentences, min_count=0, workers=16, iter=3, compute_loss=True, size=100)
print('loss:', model.get_latest_training_loss())

#model.save('../data/word2vec.model')


loss = model.get_latest_training_loss()
final_model = model
for i in range(7):
    model.train(sentences, total_examples=model.corpus_count, compute_loss=True, epochs=model.epochs)
    nloss = model.get_latest_training_loss()
    if nloss < loss:
        loss = nloss
        final_model = copy.deepcopy(model)
        print('nloss:', nloss)

print('final loss:', loss)
final_model.save('../data2/word2vec.model')

